include("./integrators3.jl")

"""
Classical Lie-Trotter-Godunov operator splitting in time.
"""
struct LTGOSSolver{AST,BST} <: AbstractSolver
    A_solver::AST
    B_solver::BST
end

"""
Caches for the classical Lie-Trotter-Godunov operator splitting scheme.
"""
struct LTGOSSolverCache{ASCT, BSCT} <: AbstractSolver
    A_solver_cache::ASCT
    B_solver_cache::BSCT
end

function setup_solver_cache(problem::SplitProblem, solver::LTGOSSolver, t₀)
    return LTGOSSolverCache(
        setup_solver_cache(problem.A, solver.A_solver, t₀),
        setup_solver_cache(problem.B, solver.B_solver, t₀),
    )
end

# Lie-Trotter-Godunov step to advance the problem split into A and B from a given initial condition
function perform_step!(problem::SplitProblem, cache::LTGOSSolverCache, t, Δt)
    # We start by setting the initial condition for the step of problem A from the solution in B.
    @timeit_debug "transfer B->A" transfer_fields!(problem.B, cache.B_solver_cache, problem.A, cache.A_solver_cache)
    # Then the step for A is executed
    @timeit_debug "step A" perform_step!(problem.A, cache.A_solver_cache, t, Δt) || return false
    # This sets the initial condition for problem B
    @timeit_debug "transfer A->B" transfer_fields!(problem.A, cache.A_solver_cache, problem.B, cache.B_solver_cache)
    # Then the step for B is executed
    @timeit_debug "step B" perform_step!(problem.B, cache.B_solver_cache, t, Δt) || return false
    # This concludes the time step
    return true
end

function setup_solver_cache(problem::SplitProblem, solver::LTGOSSolver{<:BackwardEulerSolver,<:AbstractPointwiseSolver}, t₀)
    cache = LTGOSSolverCache(
        setup_solver_cache(problem.A, solver.A_solver, t₀),
        setup_solver_cache(problem.B, solver.B_solver, t₀),
    )
    cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
    return cache
end

function setup_solver_cache(problem::SplitProblem, solver::LTGOSSolver{<:AbstractPointwiseSolver,<:BackwardEulerSolver}, t₀)
    cache = LTGOSSolverCache(
        setup_solver_cache(problem.A, solver.A_solver, t₀),
        setup_solver_cache(problem.B, solver.B_solver, t₀),
    )
    cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
    return cache
end

"""
    transfer_fields!(A, A_cache, B, B_cache)

The entry point to prepare the field evaluation for the time step of problem B, given the solution of problem A.
The default behavior assumes that nothing has to be done, because both problems use the same unknown vectors for the shared parts.
"""
transfer_fields!(A, A_cache, B, B_cache)

transfer_fields!(::DiffEqBase.AbstractDEProblem, ::Any, ::AbstractPointwiseProblem, ::AbstractPointwiseSolverCache) = nothing
transfer_fields!(::AbstractPointwiseProblem, ::AbstractPointwiseSolverCache, ::DiffEqBase.AbstractDEProblem, ::Any) = nothing
transfer_fields!(::Thunderbolt.AbstractPointwiseProblem, ::Thunderbolt.AbstractPointwiseSolverCache, ::Thunderbolt.AbstractPointwiseProblem, ::Thunderbolt.AbstractPointwiseSolverCache) = nothing

transfer_fields!(A, A_cache, B, B_cache) = @warn "IMPLEMENT ME (transfer_fields!)" maxlog=1

function setup_initial_condition!(problem::SplitProblem, cache, initial_condition, time)
    setup_initial_condition!(problem.A, cache.A_solver_cache, initial_condition, time)
    setup_initial_condition!(problem.B, cache.B_solver_cache, initial_condition, time)
    return nothing
end

# TODO what exactly is the job here? How do we know where to write and what to iterate?
function setup_initial_condition!(problem::SplitProblem{<:TransientHeatProblem, <:AbstractPointwiseProblem}, cache, initial_condition, time)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    u₀, s₀ = initial_condition(problem, time)
    # TODO maybe we should replace n with t here
    cache.B_solver_cache.uₙ .= u₀ # Note that the vectors in the caches are connected
    # TODO maybe we should replace n with t here
    cache.B_solver_cache.sₙ .= s₀
    return nothing
end

perform_step!(problem::PointwiseODEProblem, cache::AbstractPointwiseSolverCache, t, Δt) = perform_step!(problem.ode, t, Δt, cache)



# TODO add guidance with helpers like
#   const QuGarfinkel1999Solver = SMOSSolver{AdaptiveForwardEulerReactionSubCellSolver, ImplicitEulerHeatSolver}

# Transfer pressure solution from structure to lumped circuit
function transfer_fields!(A::RSAFDQ20223DProblem, A_cache::AbstractSolver, B::ODEProblem, B_cache::AbstractSolver)
    @unpack tying_cache = A_cache.inner_solver_cache.op # op = RSAFDQ20223DOperator
    u_structure = A_cache.uₙ
    u_fluid = B_cache.uₙ
    for (chamber_idx,chamber) ∈ enumerate(tying_cache.chambers)
        p = u_structure[chamber.pressure_dof_index]
        B.p[chamber_idx] = p # FIXME should not be chamber_idx
    end
end

# Transfer chamber volume from lumped circuit to structure
function transfer_fields!(A::ODEProblem, A_cache::AbstractSolver, B::RSAFDQ20223DProblem, B_cache::AbstractSolver)
    @unpack tying_cache = B_cache.inner_solver_cache.op # op = RSAFDQ20223DOperator
    u_structure = B_cache.uₙ
    u_fluid = A_cache.uₙ
    for (chamber_idx,chamber) ∈ enumerate(tying_cache.chambers)
        chamber.V⁰ᴰval = u_fluid[chamber.V⁰ᴰidx]
    end
end
