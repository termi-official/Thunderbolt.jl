
"""
    ΦReggazoniSalvadorAfrica(t,tC,tR,TC,TR,THB)

Activation transient from the paper [RegSalAfrFedDedQar:2022:cem](@citet).

     t  = time
   THB  = time for a full heart beat
[tC,TC] = contraction period
[tR,TR] = relaxation period
"""
function Φ_ReggazoniSalvadorAfrica(t,tC,tR,TC,TR,THB)
    @show t
    tnow = mod(t - tC, THB)
    if 0 ≤ tnow < TC
        return 1/2 * (1+cos(π/TC * tnow)) 
    end
    tnow = mod(t - tR, THB)
    if 0 ≤ tnow < TR
        return 1/2 * (1+cos(π/TR * tnow))
    end
    return 0.0
end

elastance_ReggazoniSalvadorAfrica(t,Epass,Emax,tC,tR,TC,TR,THB) = Epass + Emax*Φ_ReggazoniSalvadorAfrica(t,tC,tR,TC,TR,THB)


"""
    ReggazoniSalvadorAfricaLumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct ReggazoniSalvadorAfricaLumpedCicuitModel{
    T1, # mmHg s mL^-1
    T2, # mL mmHg^-1
    T3, # mL
    T4, # s
    T5, # mmHg s^2 mL^-1
}
    #
    Rsysₐᵣ::T1  = T1(0.8)
    Rpulₐᵣ::T1  = T1(0.1625)
    Rsysᵥₑₙ::T1 = T1(0.26)
    Rpulᵥₑₙ::T1 = T1(0.1625)
    #
    Csysₐᵣ::T2  = T2(1.2)
    Cpulₐᵣ::T2  = T2(10.0)
    Csysᵥₑₙ::T2 = T2(60.0)
    Cpulᵥₑₙ::T2 = T2(16.0)
    #
    Lsysₐᵣ::T5  = T1(5e-3)
    Lpulₐᵣ::T5  = T1(5e-4)
    Lsysᵥₑₙ::T5 = T1(5e-4)
    Lpulᵥₑₙ::T5 = T1(5e-4)
    # Valve stuff
    Rmin::T1 = T1(0.0075)
    Rmax::T1 = T1(75000.0)
    # Passive elastance
    Epassₗₐ::T1 = T1(0.09)
    Epassᵣₐ::T1 = T1(0.07)
    Epassᵣᵥ::T1 = T1(0.05)
    # Active elastance
    Eactmaxₗₐ::T1 = T1(0.07)
    Eactmaxᵣₐ::T1 = T1(0.06)
    Eactmaxᵣᵥ::T1 = T1(0.55)
    # "Initial volume"
    V0ₗₐ::T3 = T3(4.0)
    V0ᵣₐ::T3 = T3(4.0)
    V0ᵣᵥ::T3 = T3(10.0)
    # Event timings
    tCₗₐ::T4 = T4(0.6)
    TCₗₐ::T4 = T4(0.104)
    TRₗₐ::T4 = T4(0.68)
    TRᵣₐ::T4 = T4(0.56)
    tCᵣₐ::T4 = T4(0.064)
    TCᵣₐ::T4 = T4(0.64)
    tCᵣᵥ::T4 = T4(0.0)
    TCᵣᵥ::T4 = T4(0.272)
    TRᵣᵥ::T4 = T4(0.12)
    THB::T4 = T4(0.8) # 75 beats per minute
    # Prescribed functions
    # pₑₓ::PEX = (p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) -> 0.0
    # Eₗₐ::ELA = (p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) -> elastance_ReggazoniSalvadorAfrica(t, p.Epassₗₐ, Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    # Eᵣₐ::ERA = (p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) -> elastance_ReggazoniSalvadorAfrica(t, p.Epassᵣₐ, Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    # Eᵣᵥ::ERV = (p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) -> elastance_ReggazoniSalvadorAfrica(t, p.Epassᵣᵥ, Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)
end

# ReggazoniSalvadorAfricaLumpedCicuitModel() = ReggazoniSalvadorAfricaLumpedCicuitModel{Float64,Float64,Float64,Float64,Float64}()

num_states(::ReggazoniSalvadorAfricaLumpedCicuitModel) = 12
function initial_condition!(u, p::ReggazoniSalvadorAfricaLumpedCicuitModel)
    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = p
    u .= 0.0
    u[1] = V0ₗₐ
    u[3] = V0ᵣₐ
    u[4] = V0ᵣᵥ
end

# Evaluate the right hand side of equation system (6) in the paper
# V = volume
# p = pressure
# Q = flow rates
# E = elastance
# [x]v = ventricle [x]
function lumped_driver_lv!(du, u, t, pₗᵥ, model::ReggazoniSalvadorAfricaLumpedCicuitModel)
    Vₗₐ, vₗᵥ, Vᵣₐ, Vᵣᵥ, psysₐᵣ, psysᵥₑₙ, ppulₐᵣ, ppulᵥₑₙ, Qsysₐᵣ, Qsysᵥₑₙ, Qpulₐᵣ, Qpulᵥₑₙ = u

    @unpack Rsysₐᵣ, Rpulₐᵣ, Rsysᵥₑₙ, Rpulᵥₑₙ = model
    @unpack Csysₐᵣ, Cpulₐᵣ, Csysᵥₑₙ, Cpulᵥₑₙ = model
    @unpack Lsysₐᵣ, Lpulₐᵣ, Lsysᵥₑₙ, Lpulᵥₑₙ = model
    @unpack Rmin, Rmax = model
    # @unpack Epassₗₐ, Epassᵣₐ, Epassᵣᵥ = model
    # @unpack Eactmaxₗₐ, Eactmaxᵣₐ, Eactmaxᵣᵥ = model
    # @unpack Eₗₐ, Eᵣₐ, Eᵣᵥ = model
    @inline Eₗₐ(p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) = elastance_ReggazoniSalvadorAfrica(t, p.Epassₗₐ, p.Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    @inline Eᵣₐ(p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) = elastance_ReggazoniSalvadorAfrica(t, p.Epassᵣₐ, p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eᵣᵥ(p::ReggazoniSalvadorAfricaLumpedCicuitModel,t) = elastance_ReggazoniSalvadorAfrica(t, p.Epassᵣᵥ, p.Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)

    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = model
    @unpack tCₗₐ, TCₗₐ, TRₗₐ, TRᵣₐ, tCᵣₐ, TCᵣₐ, tCᵣᵥ, TCᵣᵥ, TRᵣᵥ = model

    #pₑₓ = 0.0 # External pressure created by organs
    pₗₐ = Eₗₐ(model, t)*(Vₗₐ - V0ₗₐ)
    pᵣₐ = Eᵣₐ(model, t)*(Vᵣₐ - V0ᵣₐ)
    pᵣᵥ = Eᵣᵥ(model, t)*(Vᵣᵥ - V0ᵣᵥ)

    @inline Rᵢ(p₁, p₂, p) = p₁ < p₂ ? p.Rmin : p.Rmax # Resistance
    @inline Qᵢ(p₁, p₂, model) = (p₁ - p₂) / Rᵢ(p₁, p₂, model)
    Qₘᵥ = Qᵢ(pₗₐ, pₗᵥ, model)
    Qₐᵥ = Qᵢ(pₗᵥ, psysₐᵣ, model)
    Qₜᵥ = Qᵢ(pᵣₐ, pᵣᵥ, model)
    Qₚᵥ = Qᵢ(pᵣᵥ, ppulₐᵣ, model)

    # change in volume
    du[1] = Qpulᵥₑₙ - Qₘᵥ # LA
    du[2] = Qₘᵥ - Qₐᵥ # LV
    du[3] = Qsysᵥₑₙ - Qₜᵥ # RA
    du[4] = Qₜᵥ - Qₚᵥ # RV

    # Pressure change
    du[5] = (Qₐᵥ - Qsysₐᵣ) / Csysₐᵣ # sys ar
    du[6] = (Qsysₐᵣ - Qsysᵥₑₙ) / Csysᵥₑₙ # sys ven
    du[7] = (Qₚᵥ - Qpulₐᵣ) / Cpulₐᵣ # pul ar
    du[8] = (Qpulₐᵣ - Qpulᵥₑₙ) / Cpulᵥₑₙ # pul ven

    # Flows
    Q9 = (psysᵥₑₙ - psysₐᵣ) / Rsysₐᵣ
    du[9]  = - Rsysₐᵣ * (Qsysₐᵣ + Q9) / Lsysₐᵣ # sys ar
    Q10 = (pᵣₐ - psysᵥₑₙ) / Rsysᵥₑₙ
    du[10] = - Rsysᵥₑₙ * (Qsysᵥₑₙ + Q10) / Lsysᵥₑₙ # sys ven
    Q11 = (ppulᵥₑₙ - ppulₐᵣ) / Rpulₐᵣ
    du[11] = - Rpulₐᵣ * (Qpulₐᵣ + Q11) / Lpulₐᵣ # pul ar
    Q12 = (pₗₐ - ppulᵥₑₙ) / Rpulᵥₑₙ
    du[12] = - Rpulᵥₑₙ * (Qpulᵥₑₙ + Q12) / Lpulᵥₑₙ # sys ar
end
