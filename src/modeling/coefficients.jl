"""
"""
struct FieldCoefficient{TA,IP<:Interpolation}
    elementwise_data::TA #3d array (element_idx, base_fun_idx, dim)
    ip::IP
end

"""
"""
function evaluate_coefficient(coeff::FieldCoefficient{TA,IP}, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim,TA,IP}
    @unpack elementwise_data, ip = coeff

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{dim, Float64})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.value(ip, i, ξ) * elementwise_data[cell_id, i]
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
evaluate_coefficient(coeff::ConstantCoefficient{T}, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim,T} = coeff.val


struct ConductivityToDiffusivityCoefficient{DTC, CC, STVC}
    conductivity_tensor_coefficient::DTC
    capacitance_coefficient::CC
    χ_coefficient::STVC
end

function Thunderbolt.evaluate_coefficient(coeff::ConductivityToDiffusivityCoefficient{DTC, CC, STVC}, cell_id::Int, ξ::Vec{rdim}, t::Float64=0.0) where {DTC, CC, STVC, rdim}
    κ = evaluate_coefficient(coeff.conductivity_tensor_coefficient, cell_id, ξ)
    Cₘ = evaluate_coefficient(coeff.capacitance_coefficient, cell_id, ξ)
    χ = evaluate_coefficient(coeff.χ_coefficient, cell_id, ξ)
    return κ/(Cₘ*χ)
end
