"""
    RSAFDQ2022TyingOperator

The 3D-0D coupled operator: an ordinary [`setup_operator`](@ref) result for the volume, plus the
chamber tying terms assembled as a second pass into the same matrix and residual.

The tying pass is Ferrite's manual pattern for algebraic variables: a facet loop over an augmented
local system `[celldofs(facet); pressure dofs]`.

Only the fused `update_linearization!` is served; there is no residual-only entry point.
"""
@concrete struct RSAFDQ2022TyingOperator
    op
    chambers
    # Per chamber: the `(sdh, cache)` pairs contributing the tying terms, and the chamber's algebraic
    # dofs queried once from the closed `DofHandler`.
    tying_caches
    pressure_dofs
end

getJ(op::RSAFDQ2022TyingOperator) = getJ(op.op)

function _update_tying_subdomain_Jr(assembler, sdh, u, p, ctx, tying_cache, pressure_dofs)
    # FIXME allocator api
    ndofs_local = ndofs_per_cell(sdh) + length(pressure_dofs)
    Kₑ = zeros(ndofs_local, ndofs_local)
    rₑ = zeros(ndofs_local)
    uₑ = zeros(ndofs_local)
    dofs = zeros(Int, ndofs_local)
    dofs[(ndofs_per_cell(sdh)+1):end] .= pressure_dofs
    for facet in FacetIterator(sdh, tying_cache.facets)
        copyto!(dofs, celldofs(facet))
        uₑ .= u[dofs]
        fill!(Kₑ, 0.0)
        fill!(rₑ, 0.0)
        # FIXME use facet directly
        assemble_facet!(
            FerriteOperators.JacobianResidualRequest(Kₑ, rₑ),
            tying_cache,
            FerriteOperators.FacetArgs((u = uₑ,), facet.cc, p, ctx),
            facet.current_facet_id,
        )
        assemble!(assembler, dofs, Kₑ, rₑ)
    end
end

function FerriteOperators.update_linearization!(
    op::RSAFDQ2022TyingOperator,
    residual::AbstractVector,
    states::NamedTuple,
    p,
    ctx,
)
    (; chambers, tying_caches, pressure_dofs) = op
    J = getJ(op)

    # Pass 1: the volume, which zeroes both targets.
    update_linearization!(op.op, residual, states, p, ctx)

    # Pass 2: the chamber tying terms, on top of what pass 1 wrote.
    assembler = start_assemble(J, residual; fillzero = false)
    @timeit_debug "assemble tying" for (chamber_index, chamber) ∈ enumerate(chambers)
        for (sdh, tying_cache) in tying_caches[chamber_index]
            _update_tying_subdomain_Jr(
                assembler,
                sdh,
                states.u,
                p,
                ctx,
                tying_cache,
                pressure_dofs[chamber_index],
            )
        end
        # The chamber row is `∫_Γ V³ᴰ(u) dΓ - V⁰ᴰ`; the facet kernel writes the integral, this is the
        # solver supplied reference volume.
        residual[only(pressure_dofs[chamber_index])] -= chamber.V⁰ᴰval
    end

    return nothing
end

function FerriteOperators.evaluate!(
    op::RSAFDQ2022TyingOperator,
    residual::AbstractVector,
    states::NamedTuple,
    p,
    ctx,
)
    error(
        "The 3D-0D coupled operator has no residual-only entry point: the tying terms are assembled " *
        "through Ferrite's matrix assembler, which writes the matrix too. Use the fused " *
        "`update_linearization!`, i.e. a full Newton rather than `simplified_newton = true`.",
    )
end

# Every subdofhandler owning a cell of the facetset contributes: a boundary set may span
# several subdomains (e.g. the apex wedges next to the myocardial hexahedra).
function _find_sdhs(dh, facetset)
    sdhs = SubDofHandler[]
    for sdh in dh.subdofhandlers
        if any(facet -> facet[1] ∈ sdh.cellset, facetset)
            push!(sdhs, sdh)
        end
    end
    return sdhs
end

function setup_3D0D_coupling_integrator(sdh, chamber, integrator::NonlinearIntegrator)
    return setup_boundary_cache(
        Pressure3D0DVolumeCouplerIntegrator(
            integrator.fqrc,
            integrator.volume_model.displacement_symbol,
            chamber.pressure_symbol,
            chamber.facets,
            chamber.volume_method,
        ),
        sdh,
    )
end

function setup_3D0D_coupling_integrator(sdh, chamber, integrator::NonlinearMultiDomainIntegrator2)
    # FIXME tighter weaveing between sdh and chamber.facets
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        has_volumetric_subdomain(grid, name) || continue
        volumetric_subdomain = grid.volumetric_subdomains[name]
        for cellset in values(volumetric_subdomain.data)
            if CellIndex(first(sdh.cellset)) ∈ cellset # FIXME how to get around this fallacy here?
                return setup_boundary_cache(
                    Pressure3D0DVolumeCouplerIntegrator(
                        subintegrator.fqrc,
                        subintegrator.volume_model.displacement_symbol,
                        chamber.pressure_symbol,
                        chamber.facets,
                        chamber.volume_method,
                    ),
                    sdh,
                )
            end
        end
    end
    return FerriteOperators.EmptySurfaceElementCache()
end

"""
    _chamber_coupling(dh, displacement_symbol, chamber)

The sparsity the chamber pressure needs, as Ferrite's coupling descriptor.

`CellCoupling` over the whole `DofHandler` is the conservative statement: every field dof may couple
to the pressure. The modelling-true statement is a `FacetCoupling` over the chamber surface, which is
what makes the off-diagonal support the endocardium rather than the mesh.
"""
_chamber_coupling(dh, displacement_symbol, chamber) = CellCoupling(
    collect(Int, Iterators.flatten(sdh.cellset for sdh in dh.subdofhandlers));
    algebraic_coupling = ((displacement_symbol, chamber.pressure_symbol),),
)

function setup_stage_operator(
    f::RSAFDQ20223DFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
)
    (; tying_info, structural_function) = f
    (; dh, ch, integrator) = structural_function
    chambers = tying_info.chambers

    # The tying pass writes into one dof shared by every chamber facet, which no coloring can make
    # race free, so the scheduling is sequential regardless of what the discretization asked for.
    couplings = Tuple(
        _chamber_coupling(dh, chamber.displacement_symbol, chamber) for chamber in chambers
    )
    strategy = AssemblyStrategy(
        FullAssembly(
            FerriteOperators.StandardOperatorSpecification(;
                algebraic_couplings = couplings,
                constraint_handler = ch,
            ),
        ),
        SequentialScheduling(),
        get_strategy(f).device,
    )

    op = setup_operator(strategy, integrator, dh; slots = THUNDERBOLT_STAGE_SLOTS)
    tying_caches = [
        [
            (sdh, setup_3D0D_coupling_integrator(sdh, chamber, integrator)) for
            sdh in _find_sdhs(dh, chamber.facets)
        ] for chamber in chambers
    ]
    pressure_dofs = [algebraic_dofs(dh, chamber.pressure_symbol) for chamber in chambers]

    return RSAFDQ2022TyingOperator(op, chambers, tying_caches, pressure_dofs)
end
