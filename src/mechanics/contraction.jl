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


Base.@kwdef struct ConstantStretchModel{TD} <: SteadyStateSarcomereModel
    λ::TD = 1.0
end
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ

𝓝(Ca, mp::PelceSunLangeveld1995Model) = Ca
