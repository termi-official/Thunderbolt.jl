"""
    ReactionDiffusionSplit{MODEL}

Annotation for the classical reaction-diffusion split of a given model.
"""
struct ReactionDiffusionSplit{MODEL}
    model::MODEL
end

"""
    LTGOSSolver

Classical Lie-Trotter-Godunov operator splitting in time.
"""
struct LTGOSSolver{AST,BST}
    A_solver::AST
    B_solver::BST
end

"""
    LTGOSSolverCache

Caches for the classical Lie-Trotter-Godunov operator splitting scheme.
"""
struct LTGOSSolverCache{ASCT, BSCT}
    A_solver_cache::ASCT
    B_solver_cache::BSCT
end

function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{AST,BST}, t₀) where {APT,BPT,AST,BST}
    return LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver),
        setup_solver_caches(problem.B, solver.B_solver),
    )
end

# Lie-Trotter-Godunov step to advance the problem split into A and B from a given initial condition
function perform_step!(problem::SplitProblem{APT, BPT}, cache::LTGOSSolverCache{ASCT, BSCT}, t, Δt) where {APT, BPT, ASCT, BSCT}
    # We start by setting the initial condition for the step of problem A from the solution in B.
    transfer_fields!(problem.B, cache.B_solver_cache, problem.A, cache.A_solver_cache)
    # Then the step for A is executed
    perform_step!(problem.A, cache.A_solver_cache, t, Δt) || return false
    # This sets the initial condition for problem B
    transfer_fields!(problem.A, cache.A_solver_cache, problem.B, cache.B_solver_cache)
    # Then the step for B is executed
    perform_step!(problem.B, cache.B_solver_cache, t, Δt) || return false
    # This concludes the time step
    return true
end

function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{BackwardEulerSolver,BST}, t₀) where {APT <: TransientHeatProblem,BPT,BST}
    cache = LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver, t₀),
        setup_solver_caches(problem.B, solver.B_solver, t₀),
    )
    cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
    return cache
end

function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{BackwardEulerSolver,BST}, t₀) where {APT,BPT <: TransientHeatProblem,BST}
    cache = LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver, t₀),
        setup_solver_caches(problem.B, solver.B_solver, t₀),
    )
    cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
    return cache
end

# TODO add guidance with helpers like
#   const QuGarfinkel1999Solver = SMOSSolver{AdaptiveForwardEulerReactionSubCellSolver, ImplicitEulerHeatSolver}


"""
    transfer_fields!(A, A_cache, B, B_cache)

The entry point to prepare the field evaluation for the time step of problem B, given the solution of problem A.
The default behavior assumes that nothing has to be done, because both problems use the same unknown vectors for the shared parts.
"""
transfer_fields!(A, A_cache, B, B_cache)

transfer_fields!(A, A_cache::BackwardEulerSolverCache, B, B_cache::ForwardEulerCellSolverCache) = nothing
transfer_fields!(A, A_cache::ForwardEulerCellSolverCache, B, B_cache::BackwardEulerSolverCache) = nothing

# TODO what exactly is the job here? How do we know where to write and what to iterate?
function setup_initial_condition!(problem::SplitProblem{<:Any, <:AbstractPointwiseProblem}, cache, initial_condition, time)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    u₀, s₀ = initial_condition(problem, time)
    cache.B_solver_cache.uₙ .= u₀ # Note that the vectors in the caches are connected
    cache.B_solver_cache.sₙ .= s₀
    return nothing
end

perform_step!(problem::PointwiseODEProblem{ODET}, cache::CT, t::Float64, Δt::Float64) where {ODET, CT} = perform_step!(problem.ode, t, Δt, cache)
