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

function setup_solver_caches(problem, solver::LoadDrivenSolver{IS}) where {IS}
    LoadDrivenSolverCache(
        setup_solver_caches(problem, solver.inner_solver),
        zeros(ndofs(problem.dh)),
        zeros(ndofs(problem.dh)),
    )
end

function setup_initial_condition!(problem, solver_cache::LoadDrivenSolverCache, initial_condition)
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

adapt_timestep(t, Δt, problem, solver_cache) = (t += Δt, Δt)

"""
    solve(problem, solver, Δt, time_span, initial_condition, [callback])

Main entry point for solvers in Thunderbolt.jl. The design is inspired by
DifferentialEquations.jl. We try to upstream as much content as possible to
make it available for packages.
"""
function solve(problem, solver, Δt₀, (t₀, T), initial_condition, callback::CALLBACK = (t,p,c) -> nothing) where {CALLBACK}
    solver_cache = setup_solver_caches(problem, solver)

    setup_initial_condition!(problem, solver_cache, initial_condition)

    Δt = Δt₀
    t = t₀
    while t < T
        @info t, Δt
        if !perform_step!(problem, solver_cache, t, Δt)
            return false
        end

        callback(t, problem, solver_cache)

        t, Δt = adapt_timestep(t, Δt, problem, solver_cache)
    end

    @info T
    if !perform_step!(problem, solver_cache, t, T-t)
        return false
    end

    callback(t, problem, solver_cache)

    return true
end
