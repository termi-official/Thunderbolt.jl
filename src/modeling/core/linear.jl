@doc raw"""
    LinearIntegrator

Represents the integrand a the linear form over some function space.
"""
struct LinearIntegrator{IntegrandType, QRC <: Union{<:QuadratureRuleCollection, Nothing}} <:
       AbstractLinearIntegrator
    integrand::IntegrandType
    qrc::QRC
end

function setup_element_cache(i::LinearIntegrator, sdh::SubDofHandler)
    return setup_element_cache(i.integrand, getquadraturerule(i.qrc, sdh), sdh)
end

# The eltype channel: `T` is the assembling device's `value_type`, threaded into the quadrature rule
# and on into the integrand's own cache.
function setup_element_cache(i::LinearIntegrator, sdh::SubDofHandler, ::Type{T}) where {T}
    return setup_element_cache(i.integrand, getquadraturerule(i.qrc, sdh, T), sdh, T)
end

"""
    setup_element_cache(integrand, qr, sdh, ::Type{T})

An integrand's element cache in the scalar type `T`. Falls back to the three-argument form, so an
integrand whose cache carries no `T`-dependent buffer needs no method of its own.
"""
setup_element_cache(integrand, qr, sdh::SubDofHandler, ::Type{T}) where {T} =
    setup_element_cache(integrand, qr, sdh)
