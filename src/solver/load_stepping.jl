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

function setup_solver_caches(problem, solver::LoadDrivenSolver{IS}, t₀) where {IS}
    LoadDrivenSolverCache(
        setup_solver_caches(problem, solver.inner_solver),
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
    # TODO remove these lines
    problem.t = t
    problem.Δt = Δt
    problem.u_prev .= solver_cache.uₜ₋₁

    solver_cache.uₜ₋₁ .= solver_cache.uₜ

    Ferrite.update!(problem.ch, t)
    apply!(solver_cache.uₜ, problem.ch)

    if !solve!(solver_cache.uₜ, problem, solver_cache.inner_solver_cache)
        @warn "Inner solver failed."
        return false
    end

    return true
end
