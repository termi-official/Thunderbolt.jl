"""
"""
abstract type SteadyStateSarcomereModel <: SteadyStateInternalVariable end

"""
TODO citation pelce paper

TODO remove explicit calcium field dependence
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
struct PelceSunLangeveld1995Cache{CF}
    calcium_field::CF
end

function state(model_cache::PelceSunLangeveld1995Cache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_field, geometry_cache, qp, time)
end

function setup_contraction_model_cache(_, contraction_model::PelceSunLangeveld1995Model)
    return PelceSunLangeveld1995Cache(contraction_model.calcium_field)
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

struct ConstantStretchCache{CF}
    calcium_field::CF
end

function state(model_cache::ConstantStretchCache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_field, geometry_cache, qp, time)
end

function setup_contraction_model_cache(_, contraction_model::ConstantStretchModel)
    return ConstantStretchCache(contraction_model.calcium_field)
end

update_contraction_model_cache!(cache::ConstantStretchCache, time, cell, cv) = nothing

𝓝(Ca, mp::ConstantStretchModel) = Ca

"""
Mean-field variant of the sarcomere model presented by Reggazoni, Dede and Quarteroni (2020).

The default parameters for the model are taken from the same paper for human cardiomyocytes at body temperature.
"""
Base.@kwdef struct RDQ20MFModel{TD, TSL0, TCF, TSL, TSV}
    # Geoemtric parameters
    LA::TD = 1.25
    LM::TD = 1.65
    LB::TD = 0.18
    SL₀::TSL0 = ConstantCoefficient(2.2)
    # RU steady state parameter
    Q::TD = 2.0
    Kd₀::TD = 0.381
    αKd::TD = -0.571
    μ::TD = 10.0
    γ::TD = 12.0
    # RU kinetics parameters
    Koff::TD = 100.0
    Kbasic::TD = 13.0
    # XB cycling paramerters
    r₀::TD = 134.31
    α::TD = 25.184
    μ₀_fP::TD = 32.653
    μ₁_fP::TD = 0.778
    # Upscaling parameter
    a_XB::TD = 22.894e3
    calcium_field::TCF
    sarcomere_length::TSL
    sarcomere_velocity::TSV
end

function initial_state!(u::AbstractVector, model::RDQ20MFModel)
    u[1] = 1.0
    u[2:end] .= 0.0
end

num_states(model::RDQ20MFModel) = 20

function rhs_fast!(dRU, uRU, x, t, p::RDQ20MFModel)
    # TODO StaticArrays
    dT_L = zeros(2,2)
    dT_R = zeros(2,2)
    ΦT_L = zeros(2,2,2,2)
    ΦT_C = zeros(2,2,2,2)
    ΦT_R = zeros(2,2,2,2)
    ΦC_C = zeros(2,2,2,2)
    dC = zeros(2,2)
    dT = zeros(2,2,2,2)

    # Initialize helper rates
    dC[2,1] = p.Koff
    dC[2,2] = p.Koff / p.μ
    for TL ∈ 1:2, TR ∈ 1:2
        permissive_neighbors = TL + TR
        dT[TL,2,TR,1] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,2,TR,2] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,1,TR,1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ;
        dT[TL,1,TR,2] = p.Q * p.Kbasic * p.γ^permissive_neighbors;
    end

    # Update RU
    ## Compute fluxes associated with center unit
    for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        ΦT_C[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dT[TL,TC,TR,CC];
        ΦC_C[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dC[CC,TC];
    end

    ## Compute rates associated with left unit
    for TL ∈ 1:2
        for TC ∈ 1:2
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
    end

    ## Compute rates associated with right unit
    for TR ∈ 1:2
        for TC ∈ 1:2
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
    end

    ## Compute fluxes associated with external units
    for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        ΦT_L[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dT_L[TC,TL]
        ΦT_R[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dT_R[TC,TR]
    end

    # Store RU rates
    for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
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

    rhs_fast!(dRU, uRU, x, t, p::RDQ20MFModel)

    # Velocity
    dSL_dt = 0.0 # TODO how to pass...?
    v = -dSL_dt / evaluate_coefficient(p.SL₀, nothing, QuadraturePoint(1, x), t)

    permissivity = 0.0
    flux_PN = 0.0
    flux_NP = 0.0

    # TODO StaticArrays
    dT = zeros(2,2,2,2)
    XB_A = zeros(4,4)

    for TL ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        permissivity += uRU[TL,2,TR,CC]
        flux_PN += uRU[TL,2,TR,CC] * dT[TL,2,TR,CC]
        flux_NP += uRU[TL,1,TR,CC] * dT[TL,1,TR,CC]
    end

    k_PN = 0.0
    k_NP = 0.0
    if permissivity >= 1e-12
        k_PN = flux_PN / permissivity
    end
    if 1.0 - permissivity >= 1e-12
        k_NP = flux_NP / (1 - permissivity)
    end

    r = p.r₀ + p.α*abs(v)
    diag_P = r + k_PN
    diag_N = r + k_NP

    XB_A[0+1,1+0] = -diag_P;
    XB_A[1+1,1+1] = -diag_P;
    XB_A[2+1,1+2] = -diag_N;
    XB_A[3+1,1+3] = -diag_N;
    XB_A[0+1,1+2] = k_NP;
    XB_A[1+1,1+3] = k_NP;
    XB_A[2+1,1+0] = k_PN;
    XB_A[3+1,1+1] = k_PN;
    XB_A[1+1,1+0] = -v;
    XB_A[3+1,1+2] = -v;

    dXB .= XB_A*uXB
    dXB[1] += p.μ₀_fP * permissivity
    dXB[2] += p.μ₁_fP * permissivity
end
