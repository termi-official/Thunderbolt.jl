"""
    ChamberBalanceCache()

Algebraic cache for the chamber rows `r[p] -= V⁰ᴰ`. One cache serves every chamber;
`args.item.index` (via [`query_cell_parameters`](@ref)) selects which chamber's solver-supplied
reference volume applies. The row is constant in `u` -- the integral half, `∫_Γ V³ᴰ(u) dΓ`, is the
facet item kernel's -- so its Jacobian block is left to the AD fallback.
"""
struct ChamberBalanceCache end

duplicate_for_device(device, cache::ChamberBalanceCache) = cache

# `V⁰ᴰ` is solver-supplied data, not element state: it arrives fresh through `p` on every sweep
# rather than through a mutable field the cache (and its per-worker duplicates) would hold a
# reference to.
FerriteOperators.query_cell_parameters(::ChamberBalanceCache, item::FerriteOperators.AlgebraicItem, p) =
    p.V⁰ᴰ[item.index]

FerriteOperators.assemble_algebraic!(
    req::FerriteOperators.ResidualRequest,
    ::ChamberBalanceCache,
    args::FerriteOperators.AlgebraicArgs,
) = (req.r[1] -= args.p; nothing)

"""
    RSAFDQ2022TyingIntegrator(volume, couplers)

A subdomain's structural integrator carrying the chamber tying terms as facet items.

The volumetric and fused boundary terms are `volume`'s. The chamber pressures are the element-local
system's global-dof tail, one per chamber in `couplers` order and the same on every subdomain, and
the endocardial facets a subdomain owns are its facet items.
"""
struct RSAFDQ2022TyingIntegrator{VI, CI} <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    volume::VI
    couplers::CI
end

FerriteOperators.setup_element_cache(
    integrator::RSAFDQ2022TyingIntegrator,
    sdh::SubDofHandler,
) = setup_element_cache(integrator.volume, sdh)

FerriteOperators.setup_boundary_cache(
    integrator::RSAFDQ2022TyingIntegrator,
    sdh::SubDofHandler,
) = setup_boundary_cache(integrator.volume, sdh)

# `global_dofs` is one declaration per subdomain, shared by the volumetric and the facet kernels, so
# every chamber's pressure is declared on every subdomain the integrator claims. That keeps chamber
# `k` at tail position `k` everywhere, which is what lets a subdomain's single facet item cache serve
# several chambers.
FerriteOperators.global_dofs(integrator::RSAFDQ2022TyingIntegrator, sdh::SubDofHandler) =
    Int[dof for coupler in integrator.couplers
        for dof in FerriteOperators.global_dofs(coupler, sdh)]

FerriteOperators.facet_items(integrator::RSAFDQ2022TyingIntegrator, sdh::SubDofHandler) =
    FacetIndex[facet for coupler in integrator.couplers
               for facet in FerriteOperators.facet_items(coupler, sdh)]

function FerriteOperators.setup_facet_item_cache(
    integrator::RSAFDQ2022TyingIntegrator,
    sdh::SubDofHandler,
)
    offset = ndofs_per_cell(sdh)
    caches = Tuple(
        setup_3D0D_coupling_cache(coupler, sdh, offset + k) for
        (k, coupler) in enumerate(integrator.couplers)
    )
    length(caches) == 1 && return only(caches)
    return ChamberTyingCache(caches, _chamber_of_facet(integrator.couplers, sdh))
end

# One item per chamber, in `couplers` order -- the same order `global_dofs` above puts them in the
# facet kernels' local-system tail, so chamber `k` here and chamber `k` there agree without a shared
# lookup. `algebraic_items` is declared once per `DofHandler`, not per subdomain, so the pressure dof
# it names does not depend on which subdomain's copy of the integrator asks.
FerriteOperators.algebraic_items(integrator::RSAFDQ2022TyingIntegrator, dh::DofHandler) =
    [[only(algebraic_dofs(dh, coupler.pressure_symbol))] for coupler in integrator.couplers]

FerriteOperators.setup_algebraic_cache(::RSAFDQ2022TyingIntegrator, ::DofHandler) = ChamberBalanceCache()

"""
    ChamberTyingCache(chamber_caches, chamber_of_facet)

The facet item cache of a subdomain whose endocardial facets belong to several chambers.

FerriteOperators admits one facet item cache per subdomain, so the chambers sharing a subdomain are
multiplexed here: `chamber_of_facet` names which of `chamber_caches` a facet belongs to.
"""
@concrete struct ChamberTyingCache <: AbstractSurfaceElementCache
    chamber_caches
    chamber_of_facet
end

duplicate_for_device(device, cache::ChamberTyingCache) = ChamberTyingCache(
    map(chamber_cache -> duplicate_for_device(device, chamber_cache), cache.chamber_caches),
    cache.chamber_of_facet,
)

FerriteOperators.provides_analytic(
    ::Type{<:ChamberTyingCache},
    kind::Union{FerriteOperators.JacobianKind{:u}, FerriteOperators.JacobianResidualKind},
) = FerriteOperators.provides_analytic(Pressure3D0DVolumeCouplerCache, kind)

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.AbstractAssemblyRequest,
    cache::ChamberTyingCache,
    args::FerriteOperators.FacetArgs,
    local_facet_index::Int,
)
    chamber = cache.chamber_of_facet[FacetIndex(cellid(args.cell), local_facet_index)]
    return assemble_facet!(req, cache.chamber_caches[chamber], args, local_facet_index)
end

function _chamber_of_facet(couplers, sdh::SubDofHandler)
    chamber_of_facet = Dict{FacetIndex, Int}()
    for (k, coupler) in enumerate(couplers), facet in FerriteOperators.facet_items(coupler, sdh)
        haskey(chamber_of_facet, facet) && error(
            "The endocardial facet $facet is declared by more than one chamber. A facet carries " *
            "the pressure of exactly one chamber.",
        )
        chamber_of_facet[facet] = k
    end
    return chamber_of_facet
end

"""
    _chamber_coupling(dh, displacement_symbol, chamber)

The sparsity the chamber pressure needs, as Ferrite's coupling descriptor.

`CellCoupling` over the whole `DofHandler` is what the shared `global_dofs` declaration requires:
the pressure sits in the tail of *every* element-local system of the subdomains carrying the tying
term, so the cell sweep scatters through the coupling entries of every cell -- even where the
element writes zeros into them. Narrowing this to the endocardial surface needs a per-item-family
`global_dofs` declaration, which FerriteOperators does not offer.
"""
_chamber_coupling(dh, displacement_symbol, chamber) = CellCoupling(
    collect(Int, Iterators.flatten(sdh.cellset for sdh in dh.subdofhandlers));
    algebraic_coupling = ((displacement_symbol, chamber.pressure_symbol),),
)

# One coupler per chamber, sharing the subintegrator's facet quadrature: the endocardium is a surface
# of the subdomain whose volumetric model that subintegrator carries.
_chamber_couplers(subintegrator, chambers) = [
    Pressure3D0DVolumeCouplerIntegrator(
        subintegrator.fqrc,
        chamber.displacement_symbol,
        chamber.pressure_symbol,
        chamber.facets,
        chamber.volume_method,
    ) for chamber in chambers
]

_tying_integrator(integrator::NonlinearIntegrator, chambers) =
    RSAFDQ2022TyingIntegrator(integrator, _chamber_couplers(integrator, chambers))

_tying_integrator(integrator::NonlinearMultiDomainIntegrator2, chambers) =
    NonlinearMultiDomainIntegrator2(
        Dict(
            name => RSAFDQ2022TyingIntegrator(
                subintegrator,
                _chamber_couplers(subintegrator, chambers),
            ) for (name, subintegrator) in integrator.subintegrators
        ),
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
    n_chambers = length(chambers)
    n_u = ndofs(dh) - n_chambers

    # The tying facets write into one dof shared by every chamber facet, which no coloring can make
    # race free, so the scheduling is sequential regardless of what the discretization asked for.
    couplings = Tuple(
        _chamber_coupling(dh, chamber.displacement_symbol, chamber) for chamber in chambers
    )
    # CSC blocks, not the CSR of FerriteOperators' own blocked-assembly example:
    # `SchurComplementLinearSolver`'s inner `UMFPACKFactorization` factorizes the (1,1) block, which
    # needs CSC.
    strategy = AssemblyStrategy(
        FullAssembly(
            FerriteOperators.BlockedOperatorSpecification(
                [n_u, n_chambers],
                BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}};
                algebraic_couplings = couplings,
                constraint_handler = ch,
            ),
        ),
        SequentialScheduling(),
        get_strategy(f).device,
    )

    return setup_operator(
        strategy,
        _tying_integrator(integrator, chambers),
        dh;
        slots = THUNDERBOLT_STAGE_SLOTS,
    )
end
