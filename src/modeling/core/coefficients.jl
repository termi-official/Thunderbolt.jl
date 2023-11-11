"""
"""
struct FieldCoefficient{TA,IP<:Interpolation}
    # TODO data structure for thos
    elementwise_data::TA #3d array (element_idx, base_fun_idx, dim)
    ip::IP
end

"""
"""
function evaluate_coefficient(coeff::FieldCoefficient, cell_cache, qp::QuadraturePoint{rdim}, t) where {rdim}
    @unpack elementwise_data, ip = coeff

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{rdim, Float64}) # TODO get correct dimension from FieldCoefficient

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.value(ip, i, qp.ξ) * elementwise_data[cellid(cell_cache), i]
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
evaluate_coefficient(coeff::ConstantCoefficient, cell_cache, qp, t) = coeff.val

struct ConductivityToDiffusivityCoefficient{DTC, CC, STVC}
    conductivity_tensor_coefficient::DTC
    capacitance_coefficient::CC
    χ_coefficient::STVC
end

function evaluate_coefficient(coeff::ConductivityToDiffusivityCoefficient, cell_cache, qp, t)
    κ  = evaluate_coefficient(coeff.conductivity_tensor_coefficient, cell_cache, qp, t)
    Cₘ = evaluate_coefficient(coeff.capacitance_coefficient, cell_cache, qp, t)
    χ  = evaluate_coefficient(coeff.χ_coefficient, cell_cache, qp, t)
    return κ/(Cₘ*χ)
end


struct AnalyticalCoefficient{F, IPG}
    f::F
    ip_g::IPG #TODO remove this
end

function evaluate_coefficient(coeff::AnalyticalCoefficient{F, <: VectorizedInterpolation{sdim}}, cell_cache, qp::QuadraturePoint, t) where {F, sdim}
    x = zero(Vec{sdim, T})
    for i in 1:getnbasefunctions(coeff.ip_g.ip)
        x += shape_value(coeff.ip_g.ip, qp.ξ, i) * cell_cache.coords[i]
    end
    return coeff.f(x, t)
end
