"""
"""
struct FieldCoefficient{TA,IP<:Interpolation}
    # TODO data structure for thos
    elementwise_data::TA #3d array (element_idx, base_fun_idx, dim)
    ip::IP
    # TODO scratch values
end

"""
"""
function evaluate_coefficient(coeff::FieldCoefficient{<:Any,ScalarInterpolation}, cell_cache, qp::QuadraturePoint{<:Any, T}, t) where T
    @unpack elementwise_data, ip = coeff

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(T)

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.shape_value(ip, qp.ξ, i) * elementwise_data[cellid(cell_cache), i]
    end
    return val / norm(val)
end

"""
"""
function evaluate_coefficient(coeff::FieldCoefficient{<:Any,VectorInterpolation{vdim}}, cell_cache, qp::QuadraturePoint{<:Any, T}, t) where {vdim,T}
    @unpack elementwise_data, ip = coeff

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{vdim, T})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.shape_value(ip, qp.ξ, i) * elementwise_data[cellid(cell_cache), i]
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

function evaluate_coefficient(coeff::AnalyticalCoefficient{F, <: VectorizedInterpolation{sdim}}, cell_cache, qp::QuadraturePoint{<:Any,T}, t) where {F, sdim, T}
    x = zero(Vec{sdim, T})
    for i in 1:getnbasefunctions(coeff.ip_g.ip)
        x += shape_value(coeff.ip_g.ip, qp.ξ, i) * cell_cache.coords[i]
    end
    return coeff.f(x, t)
end
