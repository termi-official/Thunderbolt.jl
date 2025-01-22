"""
TODO wrap the existing models into this one
"""
abstract type SteadyStateSarcomereModel <: AbstractInternalModel end

abstract type AbstractInternalMaterialStateCache end

# Material states without evolution equations. I.e. the states variables are at most a function of space (reference) and time.
abstract type TrivialInternalMaterialStateCache <: AbstractInternalMaterialStateCache end

abstract type RateIndependentMaterialStateCache <: AbstractInternalMaterialStateCache end

# Base.@kwdef struct SarcomereModel{ModelType, TCF, TSL}
#     model::ModelType
#     calcium_field::TCF
#     sarcomere_length::TSL
# end

# Base.@kwdef struct RateDependentSarcomereModel{ModelType, TCF, TSL, TSV}
#     model::ModelType
#     calcium_field::TCF
#     sarcomere_length::TSL
#     sarcomere_velocity::TSV
# end

"""
TODO citation pelce paper

TODO remove explicit calcium field dependence

!!! warning
    It should be highlighted that this model directly gives the steady state
    for the active stretch.
"""
Base.@kwdef struct PelceSunLangeveld1995Model{TD, CF} <: SteadyStateSarcomereModel
    β::TD = 3.0
    λᵃₘₐₓ::TD = 0.7
    calcium_field::CF
end

function compute_λᵃ(Ca, mp::PelceSunLangeveld1995Model)
    @unpack β, λᵃₘₐₓ = mp
    f(c) = c > 0.0 ? 0.5 + atan(β*log(c))/π  : 0.0
    return 1.0 / (1.0 + f(Ca)*(1.0/λᵃₘₐₓ - 1.0))
end

"""
"""
struct PelceSunLangeveld1995Cache{CF} <: TrivialInternalMaterialStateCache
    calcium_cache::CF
end

function state(model_cache::PelceSunLangeveld1995Cache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_cache, geometry_cache, qp, time)
end

function setup_contraction_model_cache(contraction_model::PelceSunLangeveld1995Model, qr::QuadratureRule, sdh::SubDofHandler)
    return PelceSunLangeveld1995Cache(
        setup_coefficient_cache(contraction_model.calcium_field, qr, sdh)
    )
end

update_contraction_model_cache!(cache::PelceSunLangeveld1995Cache, time, cell, cv) = nothing

𝓝(Ca, mp::PelceSunLangeveld1995Model) = Ca

"""
TODO remove explicit calcium field dependence
"""
Base.@kwdef struct ConstantStretchModel{TD, CF} <: SteadyStateSarcomereModel
    λ::TD = 1.0
    calcium_field::CF
end
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ

struct ConstantStretchCache{CF} <: TrivialInternalMaterialStateCache
    calcium_cache::CF
end

function state(model_cache::ConstantStretchCache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_cache, geometry_cache, qp, time)
end

function setup_contraction_model_cache(contraction_model::ConstantStretchModel, qr::QuadratureRule, sdh::SubDofHandler)
    return ConstantStretchCache(
        setup_coefficient_cache(contraction_model.calcium_field, qr, sdh)
    )
end

update_contraction_model_cache!(cache::ConstantStretchCache, time, cell, cv) = nothing

𝓝(Ca, mp::ConstantStretchModel) = Ca

@doc raw"""
Mean-field variant of the sarcomere model presented by [RegDedQua:2020:bdm](@citet).

The default parameters for the model are taken from the same paper for human cardiomyocytes at body temperature.

!!! note
    For this model in combination with active stress framework the assumption is
    taken that the sarcomere length is exactly $\sqrt{I_4}$.

!!! note
    In ,,Active contraction of cardiac cells: a reduced model for sarcomere
    dynamics with cooperative interactions'' there is a proposed evolution law for
    active stretches.
"""
Base.@kwdef struct RDQ20MFModel{TD}
    # Geoemtric parameters
    LA::TD = 1.25 # µm
    LM::TD = 1.65 # µm
    LB::TD = 0.18 # µm
    SL₀::TD = 2.2 # µm
    # RU steady state parameter
    Q::TD = 2.0 # unitless
    Kd₀::TD = 0.381 # µm
    αKd::TD = -0.571 # # µM/µm
    μ::TD = 10.0 # unitless
    γ::TD = 12.0 # unitless
    # RU kinetics parameters
    Koff::TD = 0.1 # 1/ms
    Kbasic::TD = 0.013 # 1/ms
    # XB cycling paramerters
    r₀::TD = 0.13431 # 1/ms
    α::TD = 25.184 # unitless
    μ₀_fP::TD = 0.032653 # 1/ms
    μ₁_fP::TD = 0.000778 # 1/ms
    # Upscaling parameter
    a_XB::TD = 22.894e3 # kPa
end

function initial_state!(u::AbstractVector, model::RDQ20MFModel)
    u[1] = 1.0
    u[2:end] .= 0.0
end

num_states(model::RDQ20MFModel) = 20

function rhs_fast!(dRU, uRU::AbstractArray{T}, x, λ, Ca, t, p::RDQ20MFModel) where T
    dT_L = @MMatrix zeros(T,2,2)
    dT_R = @MMatrix zeros(T,2,2)
    ΦT_L = @MArray zeros(T,2,2,2,2)
    ΦT_C = @MArray zeros(T,2,2,2,2)
    ΦT_R = @MArray zeros(T,2,2,2,2)
    ΦC_C = @MArray zeros(T,2,2,2,2)
    dC = @MMatrix zeros(T,2,2)
    dT = @MArray zeros(T,2,2,2,2)

    # Initialize helper rates
    sarcomere_length = p.SL₀*λ
    dC[1,1] = p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length)) * Ca
    dC[1,2] = p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length)) * Ca
    dC[2,1] = p.Koff
    dC[2,2] = p.Koff / p.μ
    @inbounds for TL ∈ 1:2, TR ∈ 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL,2,TR,1] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,2,TR,2] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,1,TR,1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ;
        dT[TL,1,TR,2] = p.Q * p.Kbasic * p.γ^permissive_neighbors;
    end

    # Update RU
    ## Compute fluxes associated with center unit
    @inbounds for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        ΦT_C[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dT[TL,TC,TR,CC]
        ΦC_C[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dC[CC,TC]
    end

    ## Compute rates associated with left unit
    @inbounds for TL ∈ 1:2, TC ∈ 1:2
        flux_tot = 0.0
        prob_tot = 0.0
        for TR ∈ 1:2, CC ∈ 1:2
            flux_tot += ΦT_C[TL,TC,TR,CC]
            prob_tot += uRU[TL,TC,TR,CC]
        end
        if prob_tot > 1e-12
            dT_L[TL,TC] = flux_tot / prob_tot
        else
            dT_L[TL,TC] = 0.0;
        end
    end

    ## Compute rates associated with right unit
    @inbounds for TR ∈ 1:2, TC ∈ 1:2
        flux_tot = 0.0
        prob_tot = 0.0
        for TL ∈ 1:2, CC ∈ 1:2
            flux_tot += ΦT_C[TL,TC,TR,CC]
            prob_tot += uRU[TL,TC,TR,CC]
        end
        if prob_tot > 1e-12
            dT_R[TR,TC] = flux_tot / prob_tot;
        else
            dT_R[TR,TC] = 0.0;
        end
    end

    ## Compute fluxes associated with external units
    @inbounds for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        ΦT_L[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dT_L[TC,TL]
        ΦT_R[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dT_R[TC,TR]
    end

    # Store RU rates
    @inbounds for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        dRU[TL,TC,TR,CC] = 
            -ΦT_L[TL,TC,TR,CC] + ΦT_L[3 - TL,TC,TR,CC] -
            ΦT_C[TL,TC,TR,CC] + ΦT_C[TL,3 - TC,TR,CC] -
            ΦT_R[TL,TC,TR,CC] + ΦT_R[TL,TC,3 - TR,CC] -
            ΦC_C[TL,TC,TR,CC] + ΦC_C[TL,TC,TR,3 - CC]
    end
end

function rhs!(du, u, x, t, p::RDQ20MFModel)
    # Direct translation from https://github.com/FrancescoRegazzoni/cardiac-activation/blob/master/models_cpp/model_RDQ20_MF.cpp
    uRU_flat = @view u[1:16]
    uRU = reshape(uRU_flat, (2,2,2,2))
    uXB = @view u[17:20]

    dRU_flat = @view du[1:16]
    dRU = reshape(dRU_flat, (2,2,2,2))
    dXB = @view du[17:20]

    rhs_fast!(dRU, uRU, x, λ, Ca,  t, p)

    permissivity = 0.0

    dT = @MArray zeros(2,2,2,2)
    @inbounds for TL ∈ 1:2, TR ∈ 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL,2,TR,1] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,2,TR,2] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,1,TR,1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ;
        dT[TL,1,TR,2] = p.Q * p.Kbasic * p.γ^permissive_neighbors;
    end

    flux_PN = 0.0
    flux_NP = 0.0
    @inbounds for TL ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        permissivity += uRU[TL,2,TR,CC]
        flux_PN += uRU[TL,2,TR,CC] * dT[TL,2,TR,CC]
        flux_NP += uRU[TL,1,TR,CC] * dT[TL,1,TR,CC]
    end

    k_PN = if permissivity >= 1e-12
        flux_PN / permissivity
    else
        0.0
    end
    k_NP = if 1.0 - permissivity >= 1e-12
        flux_NP / (1.0 - permissivity)
    else
        0.0
    end

    #           q(v) = α|v|
    r = p.r₀ + p.α*abs(dλdt)
    diag_P = r + k_PN
    diag_N = r + k_NP

    XB_A = @SMatrix [
        -diag_P      0.0     k_NP      0.0
          -dλdt  -diag_P      0.0     k_NP
           k_PN      0.0  -diag_N      0.0
            0.0     k_PN    -dλdt  -diag_N
    ];

    dXB .= XB_A*uXB
    dXB[1] += p.μ₀_fP * permissivity
    dXB[2] += p.μ₁_fP * permissivity
end

function fraction_single_overlap(model::RDQ20MFModel, λ)
    SL = λ*model.SL₀
    LMh = (model.LM - model.LB) * 0.5;
    if (SL > model.LA && SL <= model.LM)
      return (SL - model.LA) / LMh;
    elseif (SL > model.LM && SL <= 2 * model.LA - model.LB)
      return (SL + model.LM - 2 * model.LA) * 0.5 / LMh;
    elseif (SL > 2 * model.LA - model.LB && SL <= 2 * model.LA + model.LB)
      return 1.0;
    elseif (SL > 2 * model.LA + model.LB && SL <= 2 * model.LA + model.LM)
      return (model.LM + 2 * model.LA - SL) * 0.5 / LMh;
    else
      return 0.0;
    end
end

function compute_active_tension(model::RDQ20MFModel, state, sarcomere_stretch)
    model.a_XB * (state[18] + state[20]) * fraction_single_overlap(model, sarcomere_stretch)
end

function compute_active_stiffness(model::RDQ20MFModel, state, sarcomere_stretch)
    model.a_XB * (state[17] + state[19]) * fraction_single_overlap(model, sarcomere_stretch)
end
