using Thunderbolt, StaticArrays, DelimitedFiles

######################################################
mutable struct IOCallback{IO}
    io::IO
    counter::Int
end

IOCallback(io) = IOCallback(io, -1)

function (iocb::IOCallback{ParaViewWriter{PVD}})(t, problem::Thunderbolt.SplitProblem, solver_cache) where {PVD}
    # TODO local activation time
    iocb.counter += 1
    (iocb.counter % 10) == 0 || return nothing
    store_timestep!(iocb.io, t, problem.A.dh.grid)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.A_solver_cache.uₙ, :φₘ)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.B_solver_cache.sₙ[1:length(solver_cache.A_solver_cache.uₙ)], :s)
    Thunderbolt.finalize_timestep!(iocb.io, t)
    return nothing
end

function steady_state_initializer(problem::Thunderbolt.SplitProblem, t₀)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    dh = problem.A.dh
    ionic_model = problem.B.ode
    default_values = Thunderbolt.default_initial_state(ionic_model)
    u₀ = ones(ndofs(dh))*default_values[1]
    s₀ = zeros(ndofs(dh), Thunderbolt.num_states(ionic_model));
    for i ∈ 1:Thunderbolt.num_states(ionic_model)
        s₀[:, i] .= default_values[1+i]
    end
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        coordinates = getcoordinates(cell)
    end
    return u₀, s₀
end

######################################################

cs = Thunderbolt.CartesianCoordinateSystemCoefficient(Lagrange{RefHexahedron,1}()^3) # TODO normalize

κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,3}((
        κ₁, 0, 0,
           κᵣ, 0,
              κᵣ
    ))),
    Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
        AnalyticalCoefficient((x,t) -> maximum(x) < 1.5 && t < 2.0 ? 0.5 : 0.0, cs),
        [SVector((0.0, 2.1))]
    ),
    Thunderbolt.PCG2019()
)

mesh = generate_mesh(Hexahedron, (80,28,12), Vec((0.0,0.0,0.0)), Vec((20.0,7.0,3.0)))

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)

solver = LTGOSSolver(
    BackwardEulerSolver(),
    ForwardEulerCellSolver()
)

# Idea: We want to solve a semidiscrete problem, with a given compatible solver, on a time interval, with a given initial condition
# TODO iterator syntax
solve(
    problem,
    solver,
    0.01,
    (0.0, 50.0),
    steady_state_initializer,
    IOCallback(ParaViewWriter("Niederer"))
)
