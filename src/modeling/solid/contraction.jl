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
end

function state(model_cache::PelceSunLangeveld1995Cache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_field, Ferrite.cellid(geometry_cache), qp, time)
end

function setup_contraction_model_cache(_, contraction_model::PelceSunLangeveld1995Model, calcium_field)
    return PelceSunLangeveld1995Cache(calcium_field)
end

update_contraction_model_cache!(cache::PelceSunLangeveld1995Cache, time, cell, cv) = nothing

𝓝(Ca, mp::PelceSunLangeveld1995Model) = Ca

"""
"""
Base.@kwdef struct ConstantStretchModel{TD} <: SteadyStateSarcomereModel
    λ::TD = 1.0
end
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ

struct ConstantStretchCache{CF}
    calcium_field::CF
end

function state(model_cache::ConstantStretchCache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_field, Ferrite.cellid(geometry_cache), qp, time)
end

function setup_contraction_model_cache(_, contraction_model::ConstantStretchModel, calcium_field)
    return ConstantStretchCache(calcium_field)
end

update_contraction_model_cache!(cache::ConstantStretchCache, time, cell, cv) = nothing

𝓝(Ca, mp::ConstantStretchModel) = Ca

