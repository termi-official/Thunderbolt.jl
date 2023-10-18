using Thunderbolt, LinearAlgebra, SparseArrays, UnPack
import Thunderbolt: AbstractIonicModel

using TimerOutputs, BenchmarkTools


import LinearAlgebra: mul!

######################################################
using StaticArrays

mutable struct IOCallback{IO}
    io::IO
    counter::Int
end

IOCallback(io) = IOCallback(io, -1)

function (iocb::IOCallback{ParaViewWriter{PVD}})(t, problem::Thunderbolt.SplitProblem, solver_cache) where {PVD}
    iocb.counter += 1
    (iocb.counter % 10) == 0 || return nothing
    store_timestep!(iocb.io, t, problem.A.dh.grid)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.A_solver_cache.uₙ, :φₘ)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.B_solver_cache.sₙ[1:length(solver_cache.A_solver_cache.uₙ)], :s)
    Thunderbolt.finalize_timestep!(iocb.io, t)
    return nothing
end

######################################################
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
        # for (i, (x₁, x₂)) in enumerate(coordinates)
        #     if x₁ <= 1.25 && x₂ <= 1.25
        #         u₀[ϕₘ_celldofs[i]] = 50.0
        #     end
        # end
    end
    return u₀, s₀
end

function varying_tensor_field(x,t)
    λ₁ = 0.20
    λ₂ = 0.10
    λ₃ = 0.05
    f₀ = Vec((0.0, 0.0, 1.0))
    s₀ = Vec((0.0, 1.0, 0.0))
    n₀ = Vec((1.0, 0.0, 0.0))

    return λ₁ * f₀ ⊗ f₀ + λ₂ * s₀ ⊗ s₀ + λ₃ * n₀ ⊗ n₀
end

@inline ball_stim(x, x₀) = 2.0*max((1.0-2.0*norm(x-x₀)),0.0)
function stimulation_field(x,t)
    I_stim = 0.0
    stride = 20.0
    time_offset = 2.2 # manually computed from set below.
    x₀s = (
        Vec((0.0, 2.129634466864062, 0.60068891057269)),
        Vec((0.0, 3.445834971727991, 6.98429204052707)),
        Vec((0.0, 1.181293153057487, 1.88483064531099)),
        Vec((0.0, 1.825553618642536, 5.72745709362569)),
        Vec((0.0, 6.573312211894149, 1.55720004796510)),
        Vec((0.0, 7.445933657673202, 6.09786068850943)),
        Vec((0.0, 4.722803130696281, 2.62533623077152)),
        Vec((0.0, 6.081431556793001, 7.00407901380748)),
        Vec((0.0, 2.730494936539015, 3.08058965111730)),
        Vec((0.0, 4.933954050777051, 5.34778273675808)),
    )
    for x₀ ∈ x₀s
        if (norm(x₀)-time_offset)/stride ≤ t ≤ (norm(x₀)-time_offset)/stride+1.0
            I_stim += ball_stim(x, x₀)
        end
    end
    return I_stim
end

ip_geo = Lagrange{RefHexahedron,1}()^3
model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    AnalyticalCoefficient(varying_tensor_field, ip_geo),
    Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
        AnalyticalCoefficient(stimulation_field, ip_geo),
        [SVector((0.0, 1.0))]
    ),
    Thunderbolt.PCG2019()
)

mesh = generate_mesh(Hexahedron, (25, 50, 50), Vec((0.0,0.0,0.0)), Vec((4.0,8.0,8.0)))

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)

solver = LTGOSSolver(
    BackwardEulerSolver(),
    Thunderbolt.ThreadedForwardEulerCellSolver(64)
)

# Idea: We want to solve a semidiscrete problem, with a given compatible solver, on a time interval, with a given initial condition
# TODO iterator syntax
solve(
    problem,
    solver,
    0.01,
    (0.0, 15.0),
    steady_state_initializer,
    IOCallback(ParaViewWriter("transmural-wave"))
)
