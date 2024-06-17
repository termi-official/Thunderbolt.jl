using Thunderbolt

function test_solve_passive_structure(constitutive_model)
    tspan = (0.0,1.0)
    Δt = 1.0
    grid = generate_grid(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(grid, "left"), (x,t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(grid, "front"), (x,t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(grid, "bottom"), (x,t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
        Dirichlet(:d, getfacetset(grid, "right"), (x,t) -> [0.01t], [1])
    ]

    quasistaticform = semidiscretize(
        StructuralModel(:d, constitutive_model, ()),
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3),
            dbcs,
        ),
        grid
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = LoadDrivenSolver(
        NewtonRaphsonSolver(;max_iter=10)
    )
    integrator = init(problem, timestepper, dt=Δt, verbose=true)
    u₀ = copy(integrator.u)
    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀
end

@testset "Passive Structure" begin

test_solve_passive_structure(
    Thunderbolt.PK1Model(
        HolzapfelOgden2009Model(),
        Thunderbolt.EmptyInternalVariableModel(),
        ConstantCoefficient(OrthotropicMicrostructure(
            Vec((1.0, 0.0, 0.0)),
            Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)),
        )),
    )
)

end
