"""
    FieldCoefficient

A constant in time data field, interpolated per element with a given interpolation.
"""
struct FieldCoefficient{TA,IPC<:InterpolationCollection}
    # TODO data structure for this
    elementwise_data::TA #3d array (element_idx, base_fun_idx, dim)
    ip_collection::IPC
    # TODO use CellValues
end

function evaluate_coefficient(coeff::FieldCoefficient{<:Any,<:ScalarInterpolationCollection}, cell_cache, qp::QuadraturePoint{<:Any, T}, t) where T
    @unpack elementwise_data, ip_collection = coeff
    ip = getinterpolation(ip_collection, getcells(cell_cache.grid, cellid(cell_cache)))
    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(T)

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.shape_value(ip, qp.ξ, i) * elementwise_data[cellid(cell_cache), i]
    end
    return val
end

function evaluate_coefficient(coeff::FieldCoefficient{<:Any,<:VectorizedInterpolationCollection{vdim}}, cell_cache, qp::QuadraturePoint{<:Any, T}, t) where {vdim,T}
    @unpack elementwise_data, ip_collection = coeff
    ip = getinterpolation(ip_collection, getcells(cell_cache.grid, cellid(cell_cache)))
    n_base_funcs = Ferrite.getnbasefunctions(ip.ip)
    val = zero(Vec{vdim, T})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.shape_value(ip.ip, qp.ξ, i) * elementwise_data[cellid(cell_cache), i]
    end
    return val
end

"""
    ConstantCoefficient

Constant in space and time data.
"""
struct ConstantCoefficient{T}
    val::T
end

"""
"""
evaluate_coefficient(coeff::ConstantCoefficient, cell_cache, qp, t) = coeff.val

#  Internal helper for ep problems
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

struct CartesianCoordinateSystemCoefficient{IPC<:VectorizedInterpolationCollection}
    ip_collection::IPC
end

function evaluate_coefficient(coeff::CartesianCoordinateSystemCoefficient{<:VectorizedInterpolationCollection{sdim}}, cell_cache, qp::QuadraturePoint{<:Any,T}, t) where {sdim, T}
    x = zero(Vec{sdim, T})
    ip_collection = coeff.ip_collection
    ip = getinterpolation(ip_collection, getcells(cell_cache.grid, cellid(cell_cache)))
    for i in 1:getnbasefunctions(ip.ip)
        x += Ferrite.shape_value(ip.ip, qp.ξ, i) * cell_cache.coords[i]
    end
    return x
end


struct AnalyticalCoefficient{F, CSYS}
    f::F
    coordinate_system::CSYS
end

function evaluate_coefficient(coeff::AnalyticalCoefficient{F}, cell_cache, qp::QuadraturePoint{<:Any,T}, t) where {F, T}
    x = evaluate_coefficient(coeff.coordinate_system, cell_cache, qp, t)
    return coeff.f(x, t)
end

"""
    SpectralTensorCoefficient

Represent a tensor A via spectral decomposition ∑ᵢ λᵢ vᵢ ⊗ vᵢ.
"""
struct SpectralTensorCoefficient{MSC, TC}
    eigenvectors::MSC
    eigenvalues::TC
end

@inline _eval_sdt_coefficient(M::SVector{MS}, λ::SVector{λS}) where {MS, λS} = error("Incompatible dimensions! dim(M)=$MS dim(λ)=$λS")
@inline _eval_sdt_coefficient(M::SVector{rdim,<:Vec{sdim}}, λ::SVector{rdim}) where {rdim,sdim} = sum(i->λ[i] * M[i] ⊗ M[i], 1:rdim; init=zero(Tensor{2,sdim}))

function evaluate_coefficient(coeff::SpectralTensorCoefficient, cell_cache, qp::QuadraturePoint, t)
    M = evaluate_coefficient(coeff.eigenvectors, cell_cache, qp, t)
    λ = evaluate_coefficient(coeff.eigenvalues, cell_cache, qp, t)
    return _eval_sdt_coefficient(M, λ)
end

"""
    SpatiallyHomogeneousDataField

A data field which is constant in space and piecewise constant in time.

The value during the time interval [tᵢ,tᵢ₊₁] is dataᵢ, where t₀ is negative infinity and the last time point+1 is positive infinity.
"""
struct SpatiallyHomogeneousDataField{T}
    timings::Vector{Float64}
    data::Vector{T}
end

function Thunderbolt.evaluate_coefficient(coeff::SpatiallyHomogeneousDataField, cell_cache, qp::QuadraturePoint, t)
    @unpack timings, data = coeff
    i = 1
    tᵢ = timings[1]
    while tᵢ < t
        i+=1
        if i > length(timings)
            return data[end]
        end
        tᵢ = timings[i]
    end
    return data[i] # TODO interpolation
end
