"""
    ΦRegazzoniSalvadorAfrica(t,tC,tR,TC,TR,THB)

Activation transient from the paper [RegSalAfrFedDedQar:2022:cem](@citet).

     t  = time
   THB  = time for a full heart beat
[tC,TC] = contraction period
[tR,TR] = relaxation period
"""
function Φ_RegazzoniSalvadorAfrica(t,tC,tR,TC,TR,THB)
    tnow = mod(t - tC, THB)
    if 0 ≤ tnow < TC
        return 1/2 * (1-cos(π/TC * tnow)) 
    end
    tnow = mod(t - tR, THB)
    if 0 ≤ tnow < TR
        return 1/2 * (1+cos(π/TR * tnow))
    end
    return 0.0
end

elastance_RegazzoniSalvadorAfrica(t,Epass,Emax,tC,tR,TC,TR,THB) = Epass + Emax*Φ_RegazzoniSalvadorAfrica(t,tC,tR,TC,TR,THB)


"""
    RegazzoniSalvadorAfricaLumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct RegazzoniSalvadorAfricaLumpedCicuitModel{
    T1, # mmHg ms mL^-1
    T2, # mL mmHg^-1
    T3, # mL
    T4, # ms
    T5, # mmHg ms^2 mL^-1
    T6, # mmHg mL mL^-1
}
    #
    # Rsysₐᵣ::T1  = T1(0.8*1e3)
    # Rpulₐᵣ::T1  = T1(0.1625*1e3)
    # Rsysᵥₑₙ::T1 = T1(0.26*1e3)
    # Rpulᵥₑₙ::T1 = T1(0.1625*1e3)
    # #
    # Csysₐᵣ::T2  = T2(1.2)
    # Cpulₐᵣ::T2  = T2(10.0)
    # Csysᵥₑₙ::T2 = T2(60.0)
    # Cpulᵥₑₙ::T2 = T2(16.0)
    # #
    # Lsysₐᵣ::T5  = T5(5e-3*1e6)
    # Lpulₐᵣ::T5  = T5(5e-4*1e6)
    # Lsysᵥₑₙ::T5 = T5(5e-4*1e6)
    # Lpulᵥₑₙ::T5 = T5(5e-4*1e6)
    # # Valve stuff
    # Rmin::T1 = T1(0.0075*1e3)
    # Rmax::T1 = T1(75000.0*1e3)
    # # Passive elastance
    # Epassₗₐ::T6 = T6(0.09)
    # Epassᵣₐ::T6 = T6(0.07)
    # Epassᵣᵥ::T6 = T6(0.05)
    # # Active elastance
    # Eactmaxₗₐ::T6 = T6(0.07)
    # Eactmaxᵣₐ::T6 = T6(0.06)
    # Eactmaxᵣᵥ::T6 = T6(0.55)
    # # "Initial volume"
    # V0ₗₐ::T3 = T3(4.0)
    # V0ᵣₐ::T3 = T3(4.0)
    # V0ᵣᵥ::T3 = T3(10.0)
    # # Event timings
    # tCₗₐ::T4 = T4(0.6*1e3)
    # TCₗₐ::T4 = T4(0.104*1e3)
    # TRₗₐ::T4 = T4(0.68*1e3)
    # TRᵣₐ::T4 = T4(0.56*1e3)
    # tCᵣₐ::T4 = T4(0.064*1e3)
    # TCᵣₐ::T4 = T4(0.64*1e3)
    # tCᵣᵥ::T4 = T4(0.0*1e3)
    # TCᵣᵥ::T4 = T4(0.272*1e3)
    # TRᵣᵥ::T4 = T4(0.12*1e3)
    # THB::T4 = T4(0.8*1e3) # 75 beats per minute
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
    Lsysₐᵣ::T5  = T5(5e-3)
    Lpulₐᵣ::T5  = T5(5e-4)
    Lsysᵥₑₙ::T5 = T5(5e-4)
    Lpulᵥₑₙ::T5 = T5(5e-4)
    # Valve stuff
    Rmin::T1 = T1(0.0075)
    Rmax::T1 = T1(75000.0)
    # Passive elastance
    Epassₗₐ::T6 = T6(0.15)
    Epassᵣₐ::T6 = T6(0.15)
    Epassᵣᵥ::T6 = T6(0.1)
    # Active elastance
    Eactmaxₗₐ::T6 = T6(0.1)
    Eactmaxᵣₐ::T6 = T6(0.1)
    Eactmaxᵣᵥ::T6 = T6(1.4)
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
    # pₑₓ::PEX = (p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) -> 0.0
    # Eₗₐ::ELA = (p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) -> elastance_RegazzoniSalvadorAfrica(t, p.Epassₗₐ, Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    # Eᵣₐ::ERA = (p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) -> elastance_RegazzoniSalvadorAfrica(t, p.Epassᵣₐ, Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    # Eᵣᵥ::ERV = (p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) -> elastance_RegazzoniSalvadorAfrica(t, p.Epassᵣᵥ, Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)
end

# RegazzoniSalvadorAfricaLumpedCicuitModel() = RegazzoniSalvadorAfricaLumpedCicuitModel{Float64,Float64,Float64,Float64,Float64}()

num_states(::RegazzoniSalvadorAfricaLumpedCicuitModel) = 12
function initial_condition!(u, p::RegazzoniSalvadorAfricaLumpedCicuitModel)
    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = p
    u .= 0.0
    # u[1] = V0ₗₐ
    # u[3] = V0ᵣₐ
    # u[4] = V0ᵣᵥ

    u[1] = 20.0
    u[2] = 500.0
    u[3] = 20.0
    u[4] = 500.0
end

# Evaluate the right hand side of equation system (6) in the paper
# V = volume
# p = pressure
# Q = flow rates
# E = elastance
# [x]v = ventricle [x]
function lumped_driver_lv!(du, u, t, pₗᵥ, model::RegazzoniSalvadorAfricaLumpedCicuitModel)
    Vₗₐ, Vₗᵥ, Vᵣₐ, Vᵣᵥ, psysₐᵣ, psysᵥₑₙ, ppulₐᵣ, ppulᵥₑₙ, Qsysₐᵣ, Qsysᵥₑₙ, Qpulₐᵣ, Qpulᵥₑₙ = u

    @unpack Rsysₐᵣ, Rpulₐᵣ, Rsysᵥₑₙ, Rpulᵥₑₙ = model
    @unpack Csysₐᵣ, Cpulₐᵣ, Csysᵥₑₙ, Cpulᵥₑₙ = model
    @unpack Lsysₐᵣ, Lpulₐᵣ, Lsysᵥₑₙ, Lpulᵥₑₙ = model
    @unpack Rmin, Rmax = model
    # @unpack Epassₗₐ, Epassᵣₐ, Epassᵣᵥ = model
    # @unpack Eactmaxₗₐ, Eactmaxᵣₐ, Eactmaxᵣᵥ = model
    # @unpack Eₗₐ, Eᵣₐ, Eᵣᵥ = model
    # Note tR = tC+TC
    @inline Eₗᵥ(p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) = elastance_RegazzoniSalvadorAfrica(t, p.Epassᵣₐ, 2*p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eₗₐ(p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) = elastance_RegazzoniSalvadorAfrica(t, p.Epassₗₐ, p.Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    @inline Eᵣₐ(p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) = elastance_RegazzoniSalvadorAfrica(t, p.Epassᵣₐ, p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eᵣᵥ(p::RegazzoniSalvadorAfricaLumpedCicuitModel,t) = elastance_RegazzoniSalvadorAfrica(t, p.Epassᵣᵥ, p.Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)

    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = model
    @unpack tCₗₐ, TCₗₐ, TRₗₐ, TRᵣₐ, tCᵣₐ, TCᵣₐ, tCᵣᵥ, TCᵣᵥ, TRᵣᵥ = model

    #pₑₓ = 0.0 # External pressure created by organs
    pₗᵥ = Eₗᵥ(model, t)*(Vₗᵥ - 5.0)
    pₗₐ = Eₗₐ(model, t)*(Vₗₐ - V0ₗₐ)
    pᵣₐ = Eᵣₐ(model, t)*(Vᵣₐ - V0ᵣₐ)
    pᵣᵥ = Eᵣᵥ(model, t)*(Vᵣᵥ - V0ᵣᵥ)

    @inline Rᵢ(p₁, p₂, p) = p₁ < p₂ ? p.Rmin : p.Rmax # Resistance
    @inline Qᵢ(p₁, p₂, model) = (p₁ - p₂) / Rᵢ(p₁, p₂, model)
    # @inline Qᵢ(p₁, p₂, model) = max(p₁ - p₂, 0.0)
    Qₘᵥ = Qᵢ(pₗₐ, pₗᵥ, model)
    Qₐᵥ = Qᵢ(pₗᵥ, psysₐᵣ, model)
    Qₜᵥ = Qᵢ(pᵣₐ, pᵣᵥ, model)
    Qₚᵥ = Qᵢ(pᵣᵥ, ppulₐᵣ, model)
    # @show t, Qpulᵥₑₙ, Qₘᵥ, Qₐᵥ
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
    du[9]  = - Rsysₐᵣ/Lsysₐᵣ* (Qsysₐᵣ + Q9) # sys ar
    Q10 = (pᵣₐ - psysᵥₑₙ) / Rsysᵥₑₙ
    du[10] = - Rsysᵥₑₙ/Lsysᵥₑₙ * (Qsysᵥₑₙ + Q10) # sys ven
    Q11 = (ppulᵥₑₙ - ppulₐᵣ) / Rpulₐᵣ
    du[11] = - Rpulₐᵣ/Lpulₐᵣ * (Qpulₐᵣ + Q11) # pul ar
    Q12 = (pₗₐ - ppulᵥₑₙ) / Rpulᵥₑₙ
    du[12] = - Rpulᵥₑₙ/Lpulᵥₑₙ * (Qpulᵥₑₙ + Q12) # sys ar
end
