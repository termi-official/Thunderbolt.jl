using CirculatorySystemModels.OrdinaryDiffEq, ModelingToolkit, CirculatorySystemModels, SymbolicIndexingInterface
using GLMakie

"""
"""
@connector PressureConnector begin
    p(t)
end

"""
"""
@connector BaroreflexSignal begin
    B(t)
end

"""
"""
@connector ControllerMap begin
    M(t)
end

"""
    BaroreflexController(;name, k_control, M_base, M_symp, M_para)
"""
@component function BaroreflexController(;name, k_control, M_base, M_symp, M_para)
    @named in = BaroreflexSignal()
    @named out = ControllerMap()

    sts = @variables B(t), M(t)
    ps  = @parameters k_control = k_control M_base = M_base M_symp = M_symp M_para = M_para

    D = Differential(t)

    eqs = [
        D(B) ~ -k_control*(in.B - 0.5) * B * (B ≥ 0.5) - k_control*(in.B - 0.5) * (1-B) * (B < 0.5),
        M ~ M_base + (B - 0.5) * (M_symp - M_base) * (B ≥ 0.5) / 2 + (B - 0.5) * (M_para - M_base) * (B < 0.5) / 2,
        out.M ~ M
    ]

    compose(ODESystem(eqs, t, sts, ps; name=name), in, out)
end

"""
    Baroreflex(;name, p_set = 90.0, k_drive = 10.0, S = 0.02)
"""
@component function Baroreflex(;name, p_set = 90.0, k_drive = 10.0, S = 0.02)
    @named in = PressureConnector()
    @named out = BaroreflexSignal()

    # B_a ~ Normalized afferent baroreceptor signal
    # B_b ~ Balance signal
    sts = @variables B_a(t) B_b(t) = 0.5
    ps = @parameters p_set = p_set k_drive = k_drive S = S

    D = Differential(t)

    eqs = Equation[
        B_a ~ 1/(1+exp(-S*(in.p - p_set))),
        D(B_b) ~ -k_drive*(B_a - 0.5) * B_b * (B_a ≥ 0.5) - k_drive*(B_a - 0.5) * (1-B_b) * (B_a < 0.5),
        out.B ~ B_b
    ]

    compose(ODESystem(eqs, t, sts, ps; name=name), in, out)
end

"""
    LeakyResistorDiode(;name, Rₘᵢₙ, Rₘₐₓ)`

Implements the resistance across a valve following Ohm's law exhibiting diode like behaviour.

Parameters are in the cm, g, s system.
Pressure in mmHg.
Flow in cm^3/s (ml/s)

Named parameters:

`Rₘᵢₙ`     Resistance across the valve in mmHg*s/ml
`Rₘₐₓ`     Resistance across the valve in mmHg*s/ml
"""
@component function LeakyResistorDiode(;name, Rₘᵢₙ, Rₘₐₓ)
    @named oneport = CirculatorySystemModels.OnePort()
    @unpack Δp, q = oneport
    ps = @parameters Rₘᵢₙ = Rₘᵢₙ Rₘₐₓ = Rₘₐₓ
    eqs = [
            q ~ - (Δp / Rₘᵢₙ * (Δp < 0) + Δp / Rₘₐₓ * (Δp ≥ 0))
    ]
    extend(ODESystem(eqs, t, [], ps; name=name), oneport)
end

τ = 1.0

Rsysₐᵣ  = (0.8)
Rpulₐᵣ  = (0.1625)
Rsysᵥₑₙ = (0.26)
Rpulᵥₑₙ = (0.1625)
#
Csysₐᵣ  = (1.2)
Cpulₐᵣ  = (10.0)
Csysᵥₑₙ = (60.0)
Cpulᵥₑₙ = (16.0)
#
Lsysₐᵣ  = (5e-3)
Lpulₐᵣ  = (5e-4)
Lsysᵥₑₙ = (5e-4)
Lpulᵥₑₙ = (5e-4)
# Valve stuff
Rmin = (0.0075)
Rmax = (75000.0)
# Passive elastance
Epassₗₐ = (0.15)
Epassᵣₐ = (0.15)
Epassᵣᵥ = (0.1)
# Active elastance
Eactmaxₗₐ = (0.1)
Eactmaxᵣₐ = (0.1)
Eactmaxᵣᵥ = (1.4)
# "Initial volume"
V0ₗₐ = (4.0)
V0ᵣₐ = (4.0)
V0ᵣᵥ = (10.0)
# Event timings
tCₗₐ = (0.6)
TCₗₐ = (0.104)
TRₗₐ = (0.68)
TRᵣₐ = (0.56)
tCᵣₐ = (0.064)
TCᵣₐ = (0.64)
tCᵣᵥ = (0.0)
TCᵣᵥ = (0.272)
TRᵣᵥ = (0.12)
τ = (0.8)

# Extra parameters to emulate the LV
V0ₗᵥ = (5.0)
Epassₗᵥ = (0.125)
Eactmaxₗᵥ = (2.4)
TCₗᵥ = (0.30)
TRₗᵥ = (0.15)

Eshiftᵣᵥ = 0.0
Eshiftₗₐ = 1.0-tCₗₐ/τ
Eshiftᵣₐ = 1.0-tCᵣₐ/τ

LV_Vt0 = 500.0
RV_Vt0 = 500.0
LA_Vt0 = 20.0
RA_Vt0 = 20.0

## Start Modelling
@variables t

## Atria and ventricles
@named LV = ShiChamber(V₀=V0ₗᵥ, p₀=0.0, Eₘᵢₙ=Epassₗᵥ, Eₘₐₓ=Epassₗᵥ+Eactmaxₗᵥ, τ=τ, τₑₛ=TCₗᵥ, τₑₚ=TCₗᵥ+TRₗᵥ, Eshift=0.0)
@named LA = ShiChamber(V₀=V0ₗₐ, p₀=0.0, Eₘᵢₙ=Epassₗₐ, Eₘₐₓ=Epassₗₐ+Eactmaxₗₐ, τ=τ, τₑₛ=TCₗₐ, τₑₚ=TCₗₐ+TRₗₐ, Eshift=Eshiftₗₐ)
@named RV = ShiChamber(V₀=V0ᵣᵥ, p₀=0.0, Eₘᵢₙ=Epassᵣᵥ, Eₘₐₓ=Epassᵣᵥ+Eactmaxᵣᵥ, τ=τ, τₑₛ=TCᵣᵥ, τₑₚ=TCᵣᵥ+TRᵣᵥ, Eshift=0.0)
@named RA = ShiChamber(V₀=V0ᵣₐ, p₀=0.0, Eₘᵢₙ=Epassᵣₐ, Eₘₐₓ=Epassᵣₐ+Eactmaxᵣₐ, τ=τ, τₑₛ=TCᵣₐ, τₑₚ=TCₗₐ+TRₗₐ, Eshift=Eshiftᵣₐ)

## Valves as leaky diodes
@named AV = LeakyResistorDiode(Rₘᵢₙ = Rmin, Rₘₐₓ = Rmax)
@named MV = LeakyResistorDiode(Rₘᵢₙ = Rmin, Rₘₐₓ = Rmax)
@named TV = LeakyResistorDiode(Rₘᵢₙ = Rmin, Rₘₐₓ = Rmax)
@named PV = LeakyResistorDiode(Rₘᵢₙ = Rmin, Rₘₐₓ = Rmax)

####### Systemic Loop #######
# Systemic Artery ##
@named SYSAR = CRL(C=Csysₐᵣ, R=Rsysₐᵣ, L=Lsysₐᵣ)
# Systemic Vein ##
@named SYSVEN = CRL(C=Csysᵥₑₙ, R=Rsysᵥₑₙ, L=Lsysᵥₑₙ)

####### Pulmonary Loop #######
# Pulmonary Artery ##
@named PULAR = CRL(C=Cpulₐᵣ, R=Rpulₐᵣ, L=Lpulₐᵣ)
# Pulmonary Vein ##
@named PULVEN = CRL(C=Cpulᵥₑₙ, R=Rpulᵥₑₙ, L=Lpulᵥₑₙ)

##
circ_eqs = [
    connect(LV.out, AV.in)
    connect(AV.out, SYSAR.in)
    connect(SYSAR.out, SYSVEN.in)
    connect(SYSVEN.out, RA.in)
    connect(RA.out, TV.in)
    connect(TV.out, RV.in)
    connect(RV.out, PV.in)
    connect(PV.out,  PULAR.in)
    connect(PULAR.out, PULVEN.in)
    connect(PULVEN.out, LA.in)
    connect(LA.out, MV.in)
    connect(MV.out, LV.in)
]

## Compose the whole ODE system
@named _circ_model = ODESystem(circ_eqs, t)
@named circ_model = compose(_circ_model,
    [LV, RV, LA, RA, AV, MV, PV, TV, SYSAR, SYSVEN, PULAR, PULVEN])

## And simplify it
circ_sys = structural_simplify(circ_model)

## Setup ODE
u0 = [
    LV.V => LV_Vt0
    RV.V => RV_Vt0
    RA.V => RA_Vt0
    LA.V => LA_Vt0
    SYSAR.C.V => 0.0#100.0 * Csysₐᵣ
    SYSAR.L.q => 0.0
    SYSVEN.C.V => 0.0
    SYSVEN.L.q => 0.0
    PULAR.C.V => 0.0#30.0 * Cpulₐᵣ
    PULAR.L.q => 0.0
    PULVEN.C.V => 0.0
    PULVEN.L.q => 0.0
]

ps = [
    # LV.p => t -> (LV_Vt0 - V0ₗᵥ) * Epassₗᵥ
]

prob = ODEProblem(circ_sys, u0, (0.0, 20.0), ps)
##
@time RSASol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-12, saveat=19:0.01:20)

f = Figure()
axs = [
    Axis(f[1, 1], title="LV"),
    Axis(f[1, 2], title="RV"),
    Axis(f[2, 1], title="LA"),
    Axis(f[2, 2], title="RA")
]

lines!(axs[1], RSASol[LV.V], RSASol[LV.p])
lines!(axs[2], RSASol[RV.V], RSASol[RV.p])
lines!(axs[3], RSASol[LA.V], RSASol[LA.p])
lines!(axs[4], RSASol[RA.V], RSASol[RA.p])
