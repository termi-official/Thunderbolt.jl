"""
    solve(problem, solver, Δt, time_span, initial_condition, [callback])

Main entry point for solvers in Thunderbolt.jl. The design is inspired by
DifferentialEquations.jl. We try to upstream as much content as possible to
make it available for packages.

TODO iterator syntax
"""
function solve(problem, solver, Δt₀, (t₀, T), initial_condition, callback = (t,p,c) -> nothing)
    solver_cache = setup_solver_caches(problem, solver, t₀)

    setup_initial_condition!(problem, solver_cache, initial_condition, t₀)

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

adapt_timestep(t, Δt, problem, solver_cache) = (t += Δt, Δt)

"""

Main entry point to setup the initial condition of a problem for a given solver.
"""
function setup_initial_condition!(problem, cache, initial_condition, time)
    u₀ = initial_condition(problem, time)
    cache.uₙ .= u₀
    return nothing
end
