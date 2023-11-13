using Thunderbolt

@testset "Cuboid wave propagation" begin

function test_initializer(problem, t₀)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    dh = problem.A.dh
    ionic_model = problem.B.ode
    u₀ = zeros(ndofs(dh));
    s₀ = zeros(ndofs(dh), Thunderbolt.num_states(ionic_model));
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        # TODO get coordinate via coordinate_system
        coordinates = getcoordinates(cell)
        for (i, x) in enumerate(coordinates)
            u₀[ϕₘ_celldofs[i]] = norm(x)/2
        end
    end
    return u₀, s₀
end

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,3,Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5))),
    NoStimulationProtocol(),
    Thunderbolt.FHNModel()
)

mesh = generate_mesh(Hexahedron, (4, 4, 4), Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,1.0,1.0)))

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)

solver = LTGOSSolver(
    BackwardEulerSolver(),
    ForwardEulerCellSolver()
)

@test solve(
    problem,
    solver,
    1.0,
    (0.0, 10.0),
    test_initializer,
    (args...)->nothing
)
end
