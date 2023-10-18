"""
"""
struct FieldCoefficient{TA,IP<:Interpolation}
    elementwise_data::TA #3d array (element_idx, base_fun_idx, dim)
    ip::IP
end

"""
"""
function evaluate_coefficient(coeff::FieldCoefficient{TA,IP}, cell_cache, ξ::Vec{rdim, T}, t::T=T(0.0)) where {rdim,TA,IP,T}
    @unpack elementwise_data, ip = coeff

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{rdim, Float64})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.value(ip, i, ξ) * elementwise_data[cellid(cell_cache), i]
    end
    return val / norm(val)
end

"""
"""
struct ConstantCoefficient{T}
    val::T
end

"""
"""
evaluate_coefficient(coeff::ConstantCoefficient, cell_cache, ξ::Vec{dim, T}, t::T=0.0) where {dim,T} = coeff.val

struct ConductivityToDiffusivityCoefficient{DTC, CC, STVC}
    conductivity_tensor_coefficient::DTC
    capacitance_coefficient::CC
    χ_coefficient::STVC
end

function Thunderbolt.evaluate_coefficient(coeff::ConductivityToDiffusivityCoefficient{DTC, CC, STVC}, cell_cache, ξ::Vec{rdim, T}, t::T=T(0.0)) where {DTC, CC, STVC, rdim, T}
    κ = evaluate_coefficient(coeff.conductivity_tensor_coefficient, cell_cache, ξ)
    Cₘ = evaluate_coefficient(coeff.capacitance_coefficient, cell_cache, ξ)
    χ = evaluate_coefficient(coeff.χ_coefficient, cell_cache, ξ)
    return κ/(Cₘ*χ)
end


struct AnalyticalCoefficient{F, IPG}
    f::F
    ip_g::IPG #TODO remove this
end

function Thunderbolt.evaluate_coefficient(coeff::AnalyticalCoefficient{F, <: VectorizedInterpolation{sdim}}, cell_cache, ξ::Vec{rdim, T}, t::T=T(0.0)) where {F, sdim, rdim, T}
    x = zero(Vec{sdim, T})
    for i in 1:getnbasefunctions(coeff.ip_g.ip)
        x += shape_value(coeff.ip_g.ip, ξ, i) * cell_cache.coords[i]
    end
    return coeff.f(x, t)
end
