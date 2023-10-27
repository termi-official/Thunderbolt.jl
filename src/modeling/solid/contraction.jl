"""
"""
abstract type SteadyStateSarcomereModel end

"""
@TODO citation pelce paper
"""
Base.@kwdef struct PelceSunLangeveld1995Model{TD} <: SteadyStateSarcomereModel
    β::TD = 3.0
    λᵃₘₐₓ::TD = 0.7
end

function compute_λᵃ(Ca, mp::PelceSunLangeveld1995Model)
    @unpack β, λᵃₘₐₓ = mp
    f(c) = c > 0.0 ? 0.5 + atan(β*log(c))/π  : 0.0
    return 1.0 / (1.0 + f(Ca)*(1.0/λᵃₘₐₓ - 1.0))
end

"""
"""
struct PelceSunLangeveld1995Cache{CF}
    calcium_field::CF
    """
    Calcium concentration evaluated at the quadrature points.
    """
    calcium_values_qp::Vector{Float64}
end

function state(cache::PelceSunLangeveld1995Cache, qp::QuadraturePoint)
    return cache.calcium_values_qp[qp.i]
end

function setup_contraction_model_cache(cv::CV, contraction_model::PelceSunLangeveld1995Model, cf::CF) where {CV, CF}
    return PelceSunLangeveld1995Cache(cf, Vector{Float64}(undef, getnquadpoints(cv)))
end

function update_contraction_model_cache!(cache::PelceSunLangeveld1995Cache{CF}, time::Float64, cell::CellCacheType, cv::CV) where {CellCacheType, CV, CF}
    for qpᵢ ∈ 1:getnquadpoints(cv)
        ξ = cv.qr.points[qpᵢ]
        qp = QuadraturePoint(qpᵢ, ξ)
        cache.calcium_values_qp[qp.i] = evaluate_coefficient(cache.calcium_field, Ferrite.cellid(cell), qp, time)
    end
end

𝓝(Ca, mp::PelceSunLangeveld1995Model{CF}) where CF = Ca

"""
"""
Base.@kwdef struct ConstantStretchModel{TD} <: SteadyStateSarcomereModel
    λ::TD = 1.0
end
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ

struct ConstantStretchCache
end
update_contraction_model_cache!(cache::ConstantStretchCache, time::Float64, cell::CellCacheType, cv::CV) where {CellCacheType, CV} = nothing

𝓝(Ca, mp::ConstantStretchModel{TD}) where TD = Ca
