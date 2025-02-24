@doc raw"""
    NonlinearIntegrator{CoefficientType}

Represents the integrand a the nonlinear form over some function space.
"""
struct NonlinearIntegrator{VM, FM, QRC <: Union{<:QuadratureRuleCollection, Nothing}, FQRC <: Union{<:FacetQuadratureRuleCollection, Nothing}}
    volume_model::VM
    face_model::FM
    syms::Vector{Symbol}  # The symbols for all unknowns in the submodels.
    qrc::QRC
    fqrc::FQRC
end

function setup_element_cache(i::NonlinearIntegrator, sdh::SubDofHandler)
    return setup_element_cache(i.volume_model, getquadraturerule(i.qrc, sdh), sdh)
end

function setup_boundary_cache(i::NonlinearIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(i.face_model, getquadraturerule(i.fqrc, sdh), sdh)
end
