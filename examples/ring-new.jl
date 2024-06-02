# NOTE This example is work in progress. Please consult it at a later time again.
using Thunderbolt, UnPack
import Thunderbolt: OS

import Ferrite: get_grid, find_field

dt₀ = 10.0
tspan = (0.0, 1000.0)
dtvis = 25.0

function solve_test_ring(name_base, constitutive_model, grid, cs, face_models, ip_mech::Thunderbolt.VectorInterpolationCollection, qr_collection::QuadratureRuleCollection, Δt, T = 1000.0)
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    nonlinearform = semidiscretize(
        StructuralModel(:displacement, constitutive_model, face_models),
        FiniteElementDiscretization(
            Dict(:displacement => ip_mech),
            [
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
                Dirichlet(:displacement, getnodeset(grid, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
            ]
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValueCollection(qr_collection, ip_mech)
    standard_postproc = StandardMechanicalIOPostProcessor(io, cv_post, CoordinateSystemCoefficient(cs))

    # Create sparse matrix and residual vector
    solver = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100))

    Thunderbolt.legacysolve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        standard_postproc
    )
end


"""
In 'A transmurally heterogeneous orthotropic activation model for ventricular contraction and its numerical validation' it is suggested that uniform activtaion is fine.

TODO citation.

TODO add an example with a calcium profile compute via cell model and Purkinje activation
"""
calcium_profile_function(x,t) = t/1000.0 < 0.5 ? (1-x.transmural*0.7)*2.0*t/1000.0 : (2.0-2.0*t/1000.0)*(1-x.transmural*0.7)

# for (name, order, ring_grid) ∈ [
#     ("Debug-Ring", 1, Thunderbolt.generate_ring_mesh(16,4,4)),
#     # ("Linear-Ring", 1, Thunderbolt.generate_ring_mesh(40,8,8)),
#     # ("Quadratic-Ring", 2, Thunderbolt.generate_quadratic_ring_mesh(20,4,4))
# ]

name = "Debug-Ring"
order = 1
ring_grid = Thunderbolt.generate_ring_mesh(16,4,4)

qr_collection = QuadratureRuleCollection(2*order-1)

ip_fsn = LagrangeCollection{1}()^3
ip_mech = LagrangeCollection{order}()^3

ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid)

constitutive_model = ActiveStressModel(
    Guccione1991PassiveModel(),
    PiersantiActiveStress(;Tmax=10.0),
    PelceSunLangeveld1995Model(;calcium_field=AnalyticalCoefficient(
        calcium_profile_function,
        CoordinateSystemCoefficient(ring_cs)
    )),
    create_simple_microstructure_model(ring_cs, ip_fsn,
        endo_helix_angle = deg2rad(-60.0),
        epi_helix_angle = deg2rad(60.0),
        endo_transversal_angle = 0.0,
        epi_transversal_angle = 0.0,
        sheetlet_pseudo_angle = deg2rad(0)
    )
)

quasistaticform = semidiscretize(
    StructuralModel(:displacement, constitutive_model, ()),
    FiniteElementDiscretization(
        Dict(:displacement => ip_mech),
        [
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
        ]
    ),
    ring_grid
)

problem = Thunderbolt.QuasiStaticProblem(quasistaticform, tspan)
timestepper = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100))

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

using Thunderbolt.TimerOutputs
TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
end
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)
