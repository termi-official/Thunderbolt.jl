using CirculatorySystemModels.OrdinaryDiffEq, ModelingToolkit, CirculatorySystemModels, SymbolicIndexingInterface
using GLMakie

using Thunderbolt

# import Thunderbolt: PressureCouplingChamber

"""
    PressureCouplingChamber(;name)

A simple chamber model to couple the 0D and 3D models. It has a parameter `p3D` which
is the entry point for the pressure computed by the 3D model.
"""
@component function PressureCouplingChamber(;name)
    @named in = Pin()
    @named out = Pin()
    sts = @variables begin
        V(t) = 0.0#, [description = "Volume of the lumped 0D chamber"]
        p(t)
    end
    ps = @parameters begin
        p3D(t), [description = "Pressure of the associated 3D chamber"]
    end

    D = Differential(t)

    eqs = [
        0 ~ in.p - out.p
        p ~ in.p
        p ~ p3D
        D(V) ~ in.q + out.q
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

τ = 1.0e3

Rsysₐᵣ  = (0.8e3)
Rpulₐᵣ  = (0.1625e3)
Rsysᵥₑₙ = (0.26e3)
Rpulᵥₑₙ = (0.1625e3)
#
Csysₐᵣ  = (1.2)
Cpulₐᵣ  = (10.0)
Csysᵥₑₙ = (60.0)
Cpulᵥₑₙ = (16.0)
#
Lsysₐᵣ  = (5e3)
Lpulₐᵣ  = (5e2)
Lsysᵥₑₙ = (5e2)
Lpulᵥₑₙ = (5e2)
# Valve stuff
Rmin = (0.0075e3)
Rmax = (75000.0e3)
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
tCₗₐ = (0.6e3)
TCₗₐ = (0.104e3)
TRₗₐ = (0.68e3)
TRᵣₐ = (0.56e3)
tCᵣₐ = (0.064e3)
TCᵣₐ = (0.64e3)
tCᵣᵥ = (0.0e3)
TCᵣᵥ = (0.272e3)
TRᵣᵥ = (0.12e3)
# τ = (0.8e3)

# Extra parameters to emulate the LV
V0ₗᵥ = (5.0)
Epassₗᵥ = (0.125)
Eactmaxₗᵥ = (2.4)
TCₗᵥ = (0.30e3)
TRₗᵥ = (0.15e3)

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
@named LVc = PressureCouplingChamber()
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
circ_eqs_init = [
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
@named _circ_model_init = ODESystem(circ_eqs_init, t)
@named circ_model_init = compose(_circ_model_init,
    [LV, RV, LA, RA, AV, MV, PV, TV, SYSAR, SYSVEN, PULAR, PULVEN])

## And simplify it
circ_sys_init = structural_simplify(circ_model_init)

## Setup ODE
u0 = [
    LV.V => LV_Vt0
    RV.V => RV_Vt0
    RA.V => RA_Vt0
    LA.V => LA_Vt0
    SYSAR.C.V => 100.0 * Csysₐᵣ
    SYSAR.L.q => 0.0
    SYSVEN.C.V => 0.0
    SYSVEN.L.q => 0.0
    PULAR.C.V => 30.0 * Cpulₐᵣ
    PULAR.L.q => 0.0
    PULVEN.C.V => 0.0
    PULVEN.L.q => 0.0
]

prob = ODEProblem(circ_sys_init, u0, (0.0, 20.0e3))
##
@time circ_sol_init = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-12, saveat=18e3:0.01e3:20e3)

f = Figure()
axs = [
    Axis(f[1, 1], title="LV"),
    Axis(f[1, 2], title="RV"),
    Axis(f[2, 1], title="LA"),
    Axis(f[2, 2], title="RA")
]

lines!(axs[1], circ_sol_init[LV.V], circ_sol_init[LV.p])
lines!(axs[2], circ_sol_init[RV.V], circ_sol_init[RV.p])
lines!(axs[3], circ_sol_init[LA.V], circ_sol_init[LA.p])
lines!(axs[4], circ_sol_init[RA.V], circ_sol_init[RA.p])

# Build actual system
circ_eqs = [
    connect(LVc.out, AV.in)
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
    connect(MV.out, LVc.in)
]

## Compose the whole ODE system
@named _circ_model = ODESystem(circ_eqs, t)
@named circ_model = compose(_circ_model,
    [LVc, RV, LA, RA, AV, MV, PV, TV, SYSAR, SYSVEN, PULAR, PULVEN])

## And simplify it
circ_sys = structural_simplify(circ_model)

# TODO how to do the transfer
u0new = copy(circ_sol_init.u[end])

function solve_ideal_lv2(name_base, constitutive_model, grid, coordinate_system, face_models, ip_mech::Thunderbolt.VectorInterpolationCollection, qr_collection::QuadratureRuleCollection, Δt, T = 1000.0)
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    solid = StructuralModel(:displacement, constitutive_model, face_models)
    fluid = MTKLumpedCicuitModel(circ_sys, u0new, [LVc.p3D])
    coupler = LumpedFluidSolidCoupler(
        [
            ChamberVolumeCoupling(
                "Endocardium",
                RSAFDQ2022SurrogateVolume(),
                LVc.V
            )
        ],
        :displacement
    )

    problem = semidiscretize(
        RSAFDQ2022Split(RSAFDQ2022Model(solid,fluid,coupler)),
        FiniteElementDiscretization(
            Dict(:displacement => ip_mech),
            # Dirichlet[],
            [
                Dirichlet(:displacement, getfaceset(grid, "Base"), (x,t) -> [0.0], [3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
            ],
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValueCollection(qr_collection, ip_mech)
    standard_postproc = StandardMechanicalIOPostProcessor2(io, cv_post, CoordinateSystemCoefficient(coordinate_system))

    # Create sparse matrix and residual vector
    solver = LTGOSSolver(
        LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100, tol=1e-2)),
        ForwardEulerSolver(100),
    )

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        standard_postproc
    )
end

scaling_factor = 2.7
LV_grid = Thunderbolt.hexahedralize(Thunderbolt.generate_ideal_lv_mesh(15,2,7; inner_radius = Float64(scaling_factor*0.7), outer_radius = Float64(scaling_factor*1.0), longitudinal_upper = Float64(scaling_factor*0.2), apex_inner = Float64(scaling_factor*1.3), apex_outer = Float64(scaling_factor*1.5)))

order = 1
intorder = max(2*order-1,2)
ip_u = LagrangeCollection{order}()^3
qr_u = QuadratureRuleCollection(intorder-1)
LV_cs = compute_LV_coordinate_system(LV_grid)
# LV_cs = compute_midmyocardial_section_coordinate_system(LV_grid)
LV_fm = create_simple_microstructure_model(LV_cs, LagrangeCollection{1}()^3, endo_helix_angle = deg2rad(-60.0), epi_helix_angle = deg2rad(70.0), endo_transversal_angle = deg2rad(10.0), epi_transversal_angle = deg2rad(-20.0))
passive_model = HolzapfelOgden2009Model(1.5806251396691438e2, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(1000.0))
# passive_model = Guccione1991PassiveModel(;C₀ = 3e3, Bᶠᶠ = 8.0, Bˢˢ = 6.0, Bⁿⁿ = 3.0, Bᶠˢ = 12.0, Bⁿˢ = 3.0, Bᶠⁿ = 3.0, mpU = SimpleCompressionPenalty(0.8e4))

# integral_bcs = (NormalSpringBC(5.0, "Epicardium"),)
integral_bcs = ()

function linear_interpolation(t,y1,y2,t1,t2)
    y1 + (t-t1) * (y2-y1)/(t2-t1)
end

function calcium_profile_function(x,t)
    ca_peak = (1-x.transmural*0.7)
    if t < 300.0
        return linear_interpolation(t,0.0,ca_peak,0.0, 300.0)
    elseif t < 500.0
        return linear_interpolation(t,ca_peak,0.0,300.0, 500.0)
    else
        return 0.0
    end
end

solve_ideal_lv2("lv_test2",
    ActiveStressModel(
        Guccione1991PassiveModel(),
        PiersantiActiveStress(;Tmax=2.0e2),
        PelceSunLangeveld1995Model(;calcium_field=AnalyticalCoefficient(
            calcium_profile_function,
            CoordinateSystemCoefficient(LV_cs)
        )),
        LV_fm,
    ), LV_grid, LV_cs,
    integral_bcs,
    ip_u, qr_u, 5.0, 1000.0
)
