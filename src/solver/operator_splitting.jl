"""
    ReactionDiffusionSplit{MODEL}

Annotation for the classical reaction-diffusion split of a given model.
"""
struct ReactionDiffusionSplit{MODEL}
    model::MODEL
end

"""
    ReggazoniSalvadorAfricaSplit{MODEL}

Annotation for the split described in the 2022 paper.
"""
struct ReggazoniSalvadorAfricaSplit{MODEL <: CoupledModel}
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

function setup_solver_caches(problem::SplitProblem, solver::LTGOSSolver, t₀)
    return LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver, t₀),
        setup_solver_caches(problem.B, solver.B_solver, t₀),
    )
end

# Lie-Trotter-Godunov step to advance the problem split into A and B from a given initial condition
function perform_step!(problem::SplitProblem, cache::LTGOSSolverCache, t, Δt)
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

function setup_solver_caches(problem::SplitProblem, solver::LTGOSSolver{<:BackwardEulerSolver,<:AbstractPointwiseSolver}, t₀)
    cache = LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver, t₀),
        setup_solver_caches(problem.B, solver.B_solver, t₀),
    )
    cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
    return cache
end

function setup_solver_caches(problem::SplitProblem, solver::LTGOSSolver{<:AbstractPointwiseSolver,<:BackwardEulerSolver}, t₀)
    cache = LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver, t₀),
        setup_solver_caches(problem.B, solver.B_solver, t₀),
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

transfer_fields!(A, A_cache::BackwardEulerSolverCache, B, B_cache::AbstractPointwiseSolverCache) = nothing
transfer_fields!(A, A_cache::AbstractPointwiseSolverCache, B, B_cache::BackwardEulerSolverCache) = nothing

transfer_fields!(A, A_cache, B, B_cache) = @warn "IMPLEMENT ME (transfer_fields!)" maxlog=1

function setup_initial_condition!(problem::SplitProblem, cache, initial_condition, time)
    setup_initial_condition!(problem.A, cache.A_solver_cache, initial_condition, time)
    setup_initial_condition!(problem.B, cache.B_solver_cache, initial_condition, time)
    return nothing
end

function setup_initial_condition!(problem::SplitProblem{<:CoupledProblem{<:Tuple{<:Any, <: NullProblem}}, <:AbstractPointwiseProblem}, cache, initial_condition, time)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    # u₀ = initial_condition(problem, time)
    cache.A_solver_cache.uₙ .= zeros(ndofs(problem.A.base_problems[1].dh)) # TODO fixme :)
    # TODO maybe we should replace n with t here
    cache.B_solver_cache.uₙ .= zeros(problem.B.npoints*num_states(problem.B.ode))
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

