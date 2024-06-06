using Thunderbolt

@testset "Cuboid wave propagation" begin

function simple_initializer!(u₀, f::GenericSplitFunction)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.dof_ranges[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    ϕ₀ = @view u₀[heat_dofrange];
    # TODO extraction these via utility functions
    dh = heatfun.dh
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        φₘ_celldofs = _celldofs[dof_range(dh, :φₘ)]
        # TODO query coordinate directly from the cell model
        coordinates = getcoordinates(cell)
        for (i, x) in zip(φₘ_celldofs,coordinates)
            ϕ₀[i] = norm(x)/2
        end
    end
end

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,3,Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5))),
    NoStimulationProtocol(),
    Thunderbolt.FHNModel()
)

mesh = generate_mesh(Hexahedron, (4, 4, 4), Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,1.0,1.0)))

odeform = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)

timestepper = LieTrotterGodunov((
    BackwardEulerSolver(),
    ForwardEulerCellSolver()
))

u₀ = zeros(Float64, OS.function_size(odeform))
simple_initializer!(u₀, odeform)

tspan = (0.0, 10.0)
problem = OperatorSplittingProblem(odeform, u₀, tspan)

integrator = DiffEqBase.init(problem, timestepper, dt=1.0, verbose=true)
DiffEqBase.solve!(integrator)
@test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
@test integrator.u ≉ u₀

end
