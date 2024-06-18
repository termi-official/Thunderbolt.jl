using Thunderbolt

struct TestCalciumHatField end
Thunderbolt.evaluate_coefficient(coeff::TestCalciumHatField, cell_cache, qp, t) = t/1000.0 < 0.5 ? 2.0*t/1000.0 : 2.0-2.0*t/1000.0

function test_solve_contractile_cuboid(constitutive_model)
    tspan = (0.0,300.0)
    Δt = 100.0
    mesh = generate_mesh(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x,t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x,t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x,t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]

    quasistaticform = semidiscretize(
        StructuralModel(:d, constitutive_model, (
            NormalSpringBC(0.0, "right"),
            ConstantPressureBC(0.0, "back"),
            PressureFieldBC(ConstantCoefficient(0.0),"top")
        )),
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3),
            dbcs,
        ),
        mesh
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
@testset "Contracting cuboid" begin
microstructure_model = ConstantCoefficient(OrthotropicMicrostructure(
    Vec((1.0, 0.0, 0.0)),
    Vec((0.0, 1.0, 0.0)),
    Vec((0.0, 0.0, 1.0)),
))

test_solve_contractile_cuboid(ExtendedHillModel(
    HolzapfelOgden2009Model(),
    ActiveMaterialAdapter(LinearSpringModel()),
    GMKActiveDeformationGradientModel(),
    PelceSunLangeveld1995Model(;calcium_field=TestCalciumHatField()),
    microstructure_model
))

test_solve_contractile_cuboid(GeneralizedHillModel(
    LinYinPassiveModel(),
    ActiveMaterialAdapter(LinYinActiveModel()),
    GMKIncompressibleActiveDeformationGradientModel(),
    PelceSunLangeveld1995Model(;calcium_field=TestCalciumHatField()),
    microstructure_model
))

test_solve_contractile_cuboid(ActiveStressModel(
    HumphreyStrumpfYinModel(),
    SimpleActiveStress(),
    PelceSunLangeveld1995Model(;calcium_field=TestCalciumHatField()),
    microstructure_model
))

end
