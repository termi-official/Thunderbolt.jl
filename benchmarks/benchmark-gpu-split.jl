# Per step cost of the monodomain operator splitting solve of the first EP tutorial, on the host
# against the device.
#
# What this measures, and why each number is here:
#
#  * wall time per outer Lie-Trotter-Godunov step, host against device, on a grid large enough that
#    the linear solve dominates. The problem is memory bound, so a modest speedup is the expected
#    outcome; the number that would be a defect is a slowdown.
#  * host allocations per step on the device arm. A device solve that keeps allocating on the host
#    every step is paying for a transfer somewhere, and the split's index sets are the usual place:
#    `@views parent.u[idxs]` uploads a `Vector{Int}` index set to the device on every access, once
#    per child per sync. The `gathered` arm below is the same problem with the heat child's index set
#    forced back to a `Vector{Int}`, which is what that costs.
#  * whether `OrdinaryDiffEqOperatorSplitting`'s `need_sync` elides the forward and backward copies
#    between parent and child. It does exactly when the child's buffer is a `SubArray` of the
#    parent's -- which is what indexing a host vector gives, and what indexing a device vector with a
#    contiguous range does *not*: `GPUArrays` derives a new `CuArray` over the same memory instead,
#    and `need_sync` cannot see the aliasing.
#
# CUDA is a weak dependency, so this runs in the GPU test environment rather than the package one:
# `julia --project=test/gpu benchmarks/benchmark-gpu-split.jl`.

using Thunderbolt
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting

import Thunderbolt: ThreadedSparseMatrixCSR
import OrdinaryDiffEqOperatorSplitting: GenericSplitFunction, need_sync

const N       = 256
const NSTEPS  = 20
const NBLOCKS = 5
const Δt      = 1.0f0

function monodomain_form(n)
    mesh = generate_mesh(Quadrilateral, (n, n), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    ep_model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5))),
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedFHNModel{Float32}(),
        CartesianCoordinateSystem(mesh),
        :φₘ,
        :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(ep_model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

# The heat index set as a `Vector{Int}`, i.e. the shape the split emitted before it reported
# contiguous runs as ranges.
gathered_indices(form) = GenericSplitFunction(
    form.functions,
    (collect(form.solution_indices[1]), form.solution_indices[2]),
    form.synchronizers,
)

function initial_condition(form)
    u₀ = create_initial_condition(form, Float32)
    setvariable!(u₀, form, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? 1.0f0 : 0.0f0
    end
    setvariable!(u₀, form, :s) do x
        x[2] ≥ 1.25 ? 0.1f0 : 0.0f0
    end
    return u₀
end

function build(form, u0, VT, SpMatType)
    timestepper = LieTrotterGodunov((
        BackwardEulerSolver(
            solution_vector_type = VT,
            system_matrix_type   = SpMatType,
            inner_solver         = KrylovJL_CG(atol = 1.0f-8, rtol = 1.0f-6),
        ),
        AdaptiveForwardEulerSubstepper(
            solution_vector_type = VT,
            reaction_threshold   = 0.1f0,
        ),
    ))
    return init(
        OperatorSplittingProblem(form, u0, (0.0f0, Float32(1000 * Δt))),
        timestepper;
        dt = Δt,
    )
end

sync(::Vector) = nothing
sync(::CuVector) = CUDA.synchronize()

"""
Minimum per step wall time over `NBLOCKS` blocks of `NSTEPS` steps, and the host allocations one
block of steps costs, divided by the step count.
"""
function measure!(integrator)
    for _ = 1:NSTEPS # warmup
        step!(integrator)
    end
    sync(integrator.u)

    best = Inf
    for _ = 1:NBLOCKS
        t0 = time_ns()
        for _ = 1:NSTEPS
            step!(integrator)
        end
        sync(integrator.u)
        best = min(best, (time_ns() - t0) / 1.0e9 / NSTEPS)
    end

    allocated = @allocated begin
        for _ = 1:NSTEPS
            step!(integrator)
        end
        sync(integrator.u)
    end

    return best, allocated / NSTEPS
end

pad(x, n) = rpad(string(x), n)

function main()
    CUDA.functional() || error("This benchmark needs a functional CUDA device.")

    form     = monodomain_form(N)
    gathered = gathered_indices(form)
    u₀       = initial_condition(form)
    println(
        "Monodomain ", N, " x ", N,
        ", ", ndofs(form.functions[1].dh), " dofs, ",
        Thunderbolt.solution_size(form), " states, Δt = ", Δt,
    )

    arms = (
        ("CPU Float32 CSR",  build(form,     copy(u₀),     Vector{Float32},   ThreadedSparseMatrixCSR{Float32, Int32})),
        ("GPU Float32 CSR",  build(form,     CuVector(u₀), CuVector{Float32}, CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32})),
        ("GPU gathered idx", build(gathered, CuVector(u₀), CuVector{Float32}, CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32})),
    )

    println("\nSynchronization elision between parent and child:")
    for (label, integrator) in arms
        parent = integrator.u
        for (i, child) in enumerate(integrator.child_subintegrators)
            slice = @views parent[integrator.child_solution_indices[i]]
            println(
                "  ", pad(label, 18), " child ", i,
                "  u::", pad(nameof(typeof(child.u)), 10),
                " slice::", pad(nameof(typeof(slice)), 10),
                " need_sync=", need_sync(child.u, slice),
            )
        end
    end

    println("\n  ", pad("arm", 18), pad("s/step", 14), pad("host B/step", 14), "speedup")
    baseline = nothing
    for (label, integrator) in arms
        t, allocs = measure!(integrator)
        baseline === nothing && (baseline = t)
        println(
            "  ", pad(label, 18),
            pad(round(t; sigdigits = 4), 14),
            pad(round(Int, allocs), 14),
            round(baseline / t; digits = 2), "x",
        )
    end
    return nothing
end

main()
