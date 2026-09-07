"""
    _subintegrator_for_subdomain(subintegrators, sdh)

The subintegrator whose *volumetric* cellset contains the first cell of `sdh`, or `nothing` if no
declared name claims the subdomain. Every routing hook resolves its subdomain through this lookup,
so the element cache, the facet item cache and the declaration hooks always reach the same verdict.
"""
function _subintegrator_for_subdomain(subintegrators::Dict{<: String}, sdh::SubDofHandler)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in subintegrators
        cellset = getcellset(grid, name)
        if first(sdh.cellset) ∈ cellset
            return subintegrator
        end
    end
    return nothing
end

struct NonlinearMultiDomainIntegrator2 <: AbstractCondensedNonlinearIntegrator
    subintegrators::Dict{<: String, <: AbstractNonlinearIntegrator}
end

function FerriteOperators.setup_element_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return EmptyVolumetricElementCache()
    return setup_element_cache(subintegrator, sdh)
end

# The declaration hooks route on the same claim as the caches: a subdomain's global dofs -- one
# declaration per item family -- and its facet items are the ones its subintegrator declares. Without
# these forwards a routed operator silently falls back to the framework defaults and drops the
# declarations.
function FerriteOperators.global_dofs(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return ()
    return FerriteOperators.global_dofs(subintegrator, sdh)
end

function FerriteOperators.facet_item_global_dofs(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return ()
    return FerriteOperators.facet_item_global_dofs(subintegrator, sdh)
end

function FerriteOperators.facet_items(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return ()
    return FerriteOperators.facet_items(subintegrator, sdh)
end

function FerriteOperators.setup_facet_item_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return EmptySurfaceElementCache()
    return FerriteOperators.setup_facet_item_cache(subintegrator, sdh)
end

# `algebraic_items` is declared once per integrator over the whole `DofHandler`, not per subdomain,
# so there is no `sdh` to route on the way the hooks above do. FerriteOperators does not forward this
# pair for its own multi-domain integrators either (`AnyMultiDomainIntegrator` in
# `elements/domain_elements.jl`), so a subintegrator that wants item rows declares them the same way
# on every subdomain it owns -- RSAFDQ's chamber tying does exactly that -- and any one subintegrator's
# declaration already speaks for the whole handler. A subdomain that declares nothing forwards to the
# framework default, `()`. This silently drops a declaration that is present on only SOME
# subintegrators; nothing in this package does that today.
function FerriteOperators.algebraic_items(
    integrator::NonlinearMultiDomainIntegrator2,
    dh::DofHandler,
)
    isempty(integrator.subintegrators) && return ()
    return FerriteOperators.algebraic_items(first(values(integrator.subintegrators)), dh)
end

FerriteOperators.setup_algebraic_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    dh::DofHandler,
) = FerriteOperators.setup_algebraic_cache(first(values(integrator.subintegrators)), dh)

struct BilinearMultiIntegrator <: AbstractBilinearIntegrator
    subintegrators::Dict{<: String, <: AbstractBilinearIntegrator}
end

function FerriteOperators.setup_element_cache(
    integrator::BilinearMultiIntegrator,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return EmptyVolumetricElementCache()
    return setup_element_cache(subintegrator, sdh)
end

struct LinearMultiIntegrator <: AbstractLinearIntegrator
    subintegrators::Dict{<: String, <: AbstractLinearIntegrator}
end

function FerriteOperators.setup_element_cache(integrator::LinearMultiIntegrator, sdh::SubDofHandler)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return EmptyVolumetricElementCache()
    return setup_element_cache(subintegrator, sdh)
end
