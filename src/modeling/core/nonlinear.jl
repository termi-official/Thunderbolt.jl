@doc raw"""
    NonlinearIntegrator

Represents the integrand a the nonlinear form over some function space.
"""
struct NonlinearIntegrator{
    VM,
    FM,
    SYMS <: Base.AbstractVecOrTuple{Symbol},
    QRC <: Union{<:QuadratureRuleCollection, Nothing},
    FQRC <: Union{<:FacetQuadratureRuleCollection, Nothing},
} <: AbstractCondensedNonlinearIntegrator
    volume_model::VM
    facet_model::FM
    syms::SYMS  # The symbols for all unknowns in the submodels.
    qrc::QRC
    fqrc::FQRC
end

function setup_element_cache(i::NonlinearIntegrator, sdh::SubDofHandler)
    return setup_element_cache(i.volume_model, getquadraturerule(i.qrc, sdh), sdh)
end

# `get_number_of_internal_dofs_per_element` dispatches on the *element cache*, since that is what
# determines how many condensed unknowns a cell carries; per cache type methods live next to the
# cache they describe. Subdomains carrying no volumetric model contribute none.
get_number_of_internal_dofs_per_element(
    integrator,
    ::EmptyVolumetricElementCache,
    sdh::SubDofHandler,
) = Iterators.repeated(0, length(sdh.cellset))

"""
    _facet_item_models(integrator)

The facet terms of `integrator`, in declaration order.

Every facet term is a facet-item term: it declares the facets it acts on and assembles as its own
work item. Everything the integrator declares beyond the cell sweep — facet-item global dofs, facet
items, algebraic items — is derived from these.
"""
_facet_item_models(integrator::NonlinearIntegrator) = _facet_model_tuple(integrator.facet_model)

# The local system of a facet item is `[celldofs(cell); the global dofs of the item models, in
# declaration order]`. The declaration is the facet-item family's own, so the subdomain's cell sweep
# is not augmented by it and the volumetric kernels see the pure field system.
FerriteOperators.facet_item_global_dofs(integrator::NonlinearIntegrator, sdh::SubDofHandler) =
    Int[dof for model in _facet_item_models(integrator)
        for dof in FerriteOperators.facet_item_global_dofs(model, sdh)]

# The UNION of the terms' declarations, sorted. Taking the union is what makes two terms supported on
# the same facet legal -- a spring and a dashpot on one surface are one item, declared once and
# assembled by both, which `CompositeFacetItemCache` below re-gates per term. Sorted for the same
# reason `resolve_facet_items` sorts: neither a facetset's iteration order nor the order the terms
# happen to sit in may decide the item order.
function FerriteOperators.facet_items(integrator::NonlinearIntegrator, sdh::SubDofHandler)
    declared = Set{FacetIndex}()
    for model in _facet_item_models(integrator),
        facet in FerriteOperators.facet_items(model, sdh)

        push!(declared, facet)
    end
    return sort!(collect(declared); by = facet -> (facet[1], facet[2]))
end

function FerriteOperators.setup_facet_item_cache(
    integrator::NonlinearIntegrator,
    sdh::SubDofHandler,
)
    models = _facet_item_models(integrator)
    isempty(models) && return EmptySurfaceElementCache()
    qr     = getquadraturerule(integrator.fqrc, sdh)
    widths = map(model -> length(FerriteOperators.facet_item_global_dofs(model, sdh)), models)
    caches = ntuple(length(models)) do k
        offset = ndofs_per_cell(sdh) + sum(widths[1:(k-1)]; init = 0)
        FerriteOperators.setup_facet_item_cache(models[k], qr, sdh, offset .+ (1:widths[k]))
    end
    length(caches) == 1 && return only(caches)
    # `CompositeFacetItemCache` re-gates the fan-out on each term's own declared set, so terms
    # supported on the *same* facet -- a spring and a dashpot on one surface, or a spring on a
    # surface the chamber tying also integrates over -- share one item and are both assembled. Every
    # term keeps a cache, declaring facets on this subdomain or not, because the offsets above are
    # positions in the tail the whole declaration order spans.
    return FerriteOperators.CompositeFacetItemCache(
        caches,
        map(model -> Set{FacetIndex}(FerriteOperators.facet_items(model, sdh)), models),
    )
end

# One item per declaring model, in declaration order -- the same order `facet_item_global_dofs` above
# puts them in the local system's tail, so the two agree without a shared lookup. `algebraic_items` is declared
# once per `DofHandler` rather than per subdomain, so the dofs it names do not depend on which
# subdomain's copy of the integrator answers.
FerriteOperators.algebraic_items(integrator::NonlinearIntegrator, dh::DofHandler) =
    Vector{Int}[item for model in _facet_item_models(integrator)
                for item in FerriteOperators.algebraic_items(model, dh)]

function FerriteOperators.setup_algebraic_cache(
    integrator::NonlinearIntegrator,
    dh::DofHandler,
)
    caches = unique(
        typeof,
        [FerriteOperators.setup_algebraic_cache(model, dh) for
         model in _facet_item_models(integrator) if
         !isempty(FerriteOperators.algebraic_items(model, dh))],
    )
    length(caches) == 1 || error(
        "FerriteOperators admits one algebraic cache per `DofHandler`, but the facet models of " *
        "this integrator ask for $(length(caches)) ($(join(typeof.(caches), ", "))). One cache " *
        "has to serve every declared item; which item a kernel stands on arrives as `args.item`.",
    )
    return only(caches)
end
