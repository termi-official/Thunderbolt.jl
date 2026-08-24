# TODO try to reproduce this via the BlockOperator
@concrete struct AssembledRSAFDQ2022Operator <: AbstractBlockOperator
    J
    # The structural operator. Its matrix is `J[Block(1,1)]`, so its sweep needs no separate target.
    inner
    # Ferrite's assembler binds a `Vector`, and the structural block of the residual is a view into
    # the blocked one, so pass 1 fills this and the result is copied into the block.
    residual_structural
    dh
    integrator
    chambers
    tying_caches
end

# Interface
function FerriteOperators.update_linearization!(
    op::AssembledRSAFDQ2022Operator,
    states::NamedTuple,
    p,
    ctx,
)
    error("Not implemented yet.")
end
function _update_tying_subdomain_Jr(assembler, sdh, u, p, ctx, tying_cache, chamber)
    # FIXME allocator api
    Kₑ = zeros(ndofs_per_cell(sdh)+1, ndofs_per_cell(sdh)+1)
    rₑ = zeros(ndofs_per_cell(sdh)+1)
    uₑ = zeros(ndofs_per_cell(sdh)+1)
    for facet in FacetIterator(sdh, tying_cache.facets)
        # FIXME loader function
        dofs = [celldofs(facet); chamber.pressure_dof_index_local]
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
    op::AssembledRSAFDQ2022Operator,
    residual_::AbstractVector,
    states::NamedTuple,
    p,
    ctx,
)
    (; J, inner, chambers, tying_caches) = op
    u_ = states.u

    bs = blocksizes(J)
    s1 = bs[1, 1][1]
    s2 = bs[2, 2][1]
    u  = BlockedVector(u_, [s1, s2])
    ud = @view u[Block(1)]

    residual  = BlockedVector(residual_, [s1, s2])
    residuald = @view residual[Block(1)]
    residualp = @view residual[Block(2)]
    fill!(residuald, 0.0)
    fill!(residualp, 0.0)

    Jpd = @view J[Block(2, 1)]
    Jdp = @view J[Block(1, 2)]
    fill!(Jpd, 0.0)
    fill!(Jdp, 0.0)

    # Pass 1: Assemble volume as usual. The inner operator's matrix IS this block, so it writes
    # straight into place.
    rd = op.residual_structural
    update_linearization!(inner, rd, merge(states, (u = ud,)), p, ctx)
    residuald .= rd
    assembler = start_assemble(J, residual; fillzero = false)

    # Pass 2: Assemble forward and backward coupling contributions
    # TODO wrap into task system as boundary integration
    @timeit_debug "assemble tying" for (chamber_index, chamber) ∈ enumerate(chambers)
        V⁰ᴰ = chamber.V⁰ᴰval
        chamber_pressure = u[chamber.pressure_dof_index_local] # We can also make this up[pressure_dof_index] with local index

        for (sdh, tying_cache) in tying_caches[chamber_index]
            _update_tying_subdomain_Jr(assembler, sdh, u, p, ctx, tying_cache, chamber)
        end

        residualp[chamber_index] -= V⁰ᴰ

        @debug "Chamber $chamber_index" chamber_pressure V⁰ᴰ
    end

    return nothing
end
function FerriteOperators.evaluate!(
    op::AssembledRSAFDQ2022Operator,
    residual_::AbstractVector,
    states::NamedTuple,
    p,
    ctx,
)
    error("Not implemented yet.")
end

getJ(op::AssembledRSAFDQ2022Operator) = op.J
getJ(op::AssembledRSAFDQ2022Operator, i::Block) = @view op.J[i]

function _find_sdhs(dh, facetset)
    facet = first(facetset)
    sdhs = SubDofHandler[]
    for sdh in dh.subdofhandlers
        if facet[1] ∈ sdh.cellset
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

function setup_stage_operator(
    f::RSAFDQ20223DFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
)
    (; tying_info, structural_function) = f
    (; dh, integrator, assembly_strategy) = structural_function

    inner = setup_operator(assembly_strategy, integrator, dh; slots = THUNDERBOLT_STAGE_SLOTS)
    tying_caches = [
        [
            (sdh, setup_3D0D_coupling_integrator(sdh, chamber, integrator)) for
            sdh in _find_sdhs(dh, chamber.facets)
        ] for chamber in tying_info.chambers
    ]

    num_chambers = length(tying_info.chambers)
    block_sizes = [ndofs(dh), num_chambers]
    total_size = sum(block_sizes)
    # First we initialize an empty dummy block array
    Jblock = BlockArray(spzeros(total_size, total_size), block_sizes, block_sizes)
    # By reference: the inner operator assembles into this block directly.
    Jblock[Block(1, 1)] = getJ(inner)
    # TODO optimize storage
    Jblock[Block(1, 2)] = sparse(ones(ndofs(dh), num_chambers))
    Jblock[Block(2, 1)] = sparse(ones(num_chambers, ndofs(dh)))
    Jblock[Block(2, 2)] = sparse(ones(num_chambers, num_chambers))
    Ferrite.fillzero!(Jblock)

    return AssembledRSAFDQ2022Operator(
        Jblock,
        inner,
        zeros(ndofs(dh)),
        dh,
        integrator,
        tying_info.chambers,
        tying_caches,
    )
end
