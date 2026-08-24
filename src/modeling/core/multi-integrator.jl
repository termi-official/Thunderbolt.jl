"""
    _subintegrator_for_subdomain(subintegrators, sdh)

The subintegrator whose *volumetric* cellset contains the first cell of `sdh`, or `nothing` if no
declared name claims the subdomain. Every routing hook resolves its subdomain through this lookup,
so element cache, boundary cache and the declaration hooks always reach the same verdict.
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

struct NonlinearMultiDomainIntegrator2 <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    subintegrators::Dict{<: String, <: AbstractNonlinearIntegrator}
end

function FerriteOperators.setup_element_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return FerriteOperators.EmptyVolumetricElementCache()
    return setup_element_cache(subintegrator, sdh)
end

# The subintegrators are keyed by *volumetric* subdomain name, so a subdomain is matched here exactly
# as in `setup_element_cache` above: the subintegrator that owns these cells also owns their weak
# boundary terms. Which facets of the subdomain actually carry a term is decided later, per facet, by
# `is_facet_in_cache`.
function FerriteOperators.setup_boundary_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return FerriteOperators.EmptySurfaceElementCache()
    return setup_boundary_cache(subintegrator, sdh)
end

# The declaration hooks route on the same claim as the caches: a subdomain's global dofs and its
# facet items are the ones its subintegrator declares. Without these forwards a routed operator
# silently falls back to the framework defaults and drops both declarations.
function FerriteOperators.global_dofs(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return ()
    return FerriteOperators.global_dofs(subintegrator, sdh)
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
    subintegrator === nothing && return FerriteOperators.EmptySurfaceElementCache()
    return FerriteOperators.setup_facet_item_cache(subintegrator, sdh)
end

struct BilinearMultiIntegrator <: AbstractBilinearIntegrator
    subintegrators::Dict{<: String, <: AbstractBilinearIntegrator}
end

function FerriteOperators.setup_element_cache(
    integrator::BilinearMultiIntegrator,
    sdh::SubDofHandler,
)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return FerriteOperators.EmptyVolumetricElementCache()
    return setup_element_cache(subintegrator, sdh)
end

function FerriteOperators.setup_boundary_cache(
    integrator::BilinearMultiIntegrator,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        has_surface_subdomain(grid, name) || continue
        surface_subdomain = grid.surface_subdomains[name]
        for facetset in values(surface_subdomain.data)
            cellset = first.(facetset)
            if first(sdh.cellset) ∈ cellset
                return setup_boundary_cache(subintegrator, sdh)
            end
        end
    end
    return FerriteOperators.EmptySurfaceElementCache()
end


struct LinearMultiIntegrator <: AbstractLinearIntegrator
    subintegrators::Dict{<: String, <: AbstractLinearIntegrator}
end

function FerriteOperators.setup_element_cache(integrator::LinearMultiIntegrator, sdh::SubDofHandler)
    subintegrator = _subintegrator_for_subdomain(integrator.subintegrators, sdh)
    subintegrator === nothing && return FerriteOperators.EmptyVolumetricElementCache()
    return setup_element_cache(subintegrator, sdh)
end

function FerriteOperators.setup_boundary_cache(
    integrator::LinearMultiIntegrator,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        has_surface_subdomain(grid, name) || continue
        surface_subdomain = grid.surface_subdomains[name]
        for facetset in values(surface_subdomain.data)
            cellset = first.(facetset)
            if first(sdh.cellset) ∈ cellset
                return setup_boundary_cache(subintegrator, sdh)
            end
        end
    end
    return FerriteOperators.EmptySurfaceElementCache()
end
