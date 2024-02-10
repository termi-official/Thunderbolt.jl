using Thunderbolt

struct TestCalciumHatField end
Thunderbolt.evaluate_coefficient(coeff::TestCalciumHatField, cell_cache, qp, t) = t/1000.0 < 0.5 ? 2.0*t/1000.0 : 2.0-2.0*t/1000.0

function test_solve_contractile_cuboid(constitutive_model)
    T  = 300.0
    Δt = 100.0
    grid = generate_grid(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))

    # Clamp three sides
    dbcs = [
        Dirichlet(:displacement, getfaceset(grid, "left"), (x,t) -> [0.0], [1])
        Dirichlet(:displacement, getfaceset(grid, "front"), (x,t) -> [0.0], [2])
        Dirichlet(:displacement, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3])
        Dirichlet(:displacement, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]

    problem = semidiscretize(
        StructuralModel(constitutive_model, [
            NormalSpringBC(0.0, "right"),
            ConstantPressureBC(0.0, "back"),
            PressureFieldBC(ConstantCoefficient(0.0),"top")
        ]),
        FiniteElementDiscretization(
            Dict(:displacement => LagrangeCollection{1}()^3),
            dbcs,
        ),
        grid
    )

    # Create sparse matrix and residual vector
    solver = LoadDrivenSolver(
        NewtonRaphsonSolver(;max_iter=10)
    )

    @test Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        (args...)->nothing
    )
end
@testset "Contracting cuboid" begin
microstructure_model = ConstantCoefficient((
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
