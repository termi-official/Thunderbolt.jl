"""
    LoadDrivenSolver{IS, T, PFUN}

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
mutable struct LoadDrivenSolver{IS}
    inner_solver::IS
end

mutable struct LoadDrivenSolverCache{ISC, T}
    inner_solver_cache::ISC
    uₜ::Vector{T}
    uₜ₋₁::Vector{T}
end

struct TimeFrozenProblem{InnerProblemType}
    inner_problem::InnerProblemType
    t::Float64
end

update_linearization!(Jᵤ, residual, u, problem::PT) where {PT <: TimeFrozenProblem} = update_linearization!(Jᵤ, residual, u, problem.inner_problem, problem.t) 
eliminate_constraints_from_linearization!(solver_cache,problem::PT) where {PT <: TimeFrozenProblem} = eliminate_constraints_from_linearization!(solver_cache, problem.inner_problem)
residual_norm(solver_cache::NewtonRaphsonSolverCache, problem::PT) where {PT <: TimeFrozenProblem} = residual_norm(solver_cache, problem.inner_problem)
eliminate_constraints_from_increment!(Δu, solver_cache, problem::PT) where {PT <: TimeFrozenProblem} = eliminate_constraints_from_increment!(Δu, solver_cache, problem.inner_problem)

function setup_solver_caches(problem, solver::LoadDrivenSolver{IS}, t₀) where {IS}
    LoadDrivenSolverCache(
        setup_solver_caches(problem, solver.inner_solver, t₀),
        zeros(ndofs(problem.dh)),
        zeros(ndofs(problem.dh)),
    )
end

function setup_initial_condition!(problem, solver_cache::LoadDrivenSolverCache, initial_condition, t₀)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    solver_cache.uₜ = zeros(ndofs(problem.dh))
    solver_cache.uₜ₋₁ = zeros(ndofs(problem.dh))
    return nothing
end

function perform_step!(problem, solver_cache::LoadDrivenSolverCache, t, Δt)
    solver_cache.uₜ₋₁ .= solver_cache.uₜ

    Ferrite.update!(problem.ch, t)
    apply!(solver_cache.uₜ, problem.ch)

    if !solve!(solver_cache.uₜ, TimeFrozenProblem(problem, t), solver_cache.inner_solver_cache)
        @warn "Inner solver failed."
        return false
    end

    return true
end
