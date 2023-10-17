using Thunderbolt, LinearAlgebra, SparseArrays, UnPack
import Thunderbolt: AbstractIonicModel

using TimerOutputs, BenchmarkTools

# using Krylov

import LinearAlgebra: mul!

using SparseArrays: AbstractSparseMatrixCSC #,AbstractSparseMatrixCSR?
# using SparseMatricesCSR #, ThreadedSparseCSR
# ThreadedSparseCSR.multithread_matmul(PolyesterThreads())

######################################################
using StaticArrays
struct PlanarDiffusionTensorCoefficient{MSC}
    microstructure_cache::MSC
    conductivities::SVector{2}
end

function Thunderbolt.evaluate_coefficient(coeff::PlanarDiffusionTensorCoefficient{MSC}, cell_id::Int, ξ::Vec{rdim}, t::Float64=0.0) where {MSC, rdim}
    f₀, s₀ = directions(coeff.microstructure_cache, cell_id, ξ, t)
    return coeff.conductivities[1] * f₀ ⊗ f₀ + coeff.conductivities[2] * s₀ ⊗ s₀
end

mutable struct IOCallback{IO}
    io::IO
end

function (iocb::IOCallback{ParaViewWriter{PVD}})(t, problem::Thunderbolt.SplitProblem, solver_cache) where {PVD}
    store_timestep!(iocb.io, t, problem.A.dh.grid)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.A_solver_cache.uₙ, :φₘ)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.B_solver_cache.sₙ[1:length(solver_cache.A_solver_cache.uₙ)], :s)
    Thunderbolt.finalize_timestep!(iocb.io, t)
end

######################################################
function spiral_wave_initializer(problem::Thunderbolt.SplitProblem, t₀)
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
        for (i, (x₁, x₂)) in enumerate(coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                u₀[ϕₘ_celldofs[i]] = 1.0
            end
            if x₂ >= 1.25
                s₀[ϕₘ_celldofs[i],1] = 0.1
            end
        end
    end
    return u₀, s₀
end

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5))),
    NoStimulationProtocol(),
    Thunderbolt.FHNModel()
)

mesh = generate_mesh(Quadrilateral, (2^6, 2^6), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

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
    1.0,
    (0.0, 1000.0),
    spiral_wave_initializer,
    IOCallback(ParaViewWriter("test"))
)
