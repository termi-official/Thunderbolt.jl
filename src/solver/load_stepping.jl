"""
    LoadDrivenSolver{IS, T, PFUN}

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
mutable struct LoadDrivenSolver{IS} <: AbstractSolver
    inner_solver::IS
end

mutable struct LoadDrivenSolverCache{ISC, T, VT <: AbstractVector{T}} <: AbstractTimeSolverCache
    inner_solver_cache::ISC
    uₙ::VT
    uₙ₋₁::VT
    tmp::VT
end

function setup_solver_cache(problem, solver::LoadDrivenSolver{<:NewtonRaphsonSolver{T}}, t₀) where T
    inner_solver_cache = setup_solver_cache(problem, solver.inner_solver)
    LoadDrivenSolverCache(
        inner_solver_cache,
        Vector{T}(undef, solution_size(problem)),
        Vector{T}(undef, solution_size(problem)),
        Vector{T}(undef, solution_size(problem)),
    )
end

setup_solver_cache(problem::AbstractCoupledProblem, solver::LoadDrivenSolver, t₀) = error("Not implemented yet.")

function setup_solver_cache(problem::AbstractCoupledProblem, solver::LoadDrivenSolver{<:NewtonRaphsonSolver{T}}, t₀) where T
    inner_solver_cache = setup_solver_cache(problem, solver.inner_solver)
    LoadDrivenSolverCache(
        inner_solver_cache,
        mortar([
            Vector{T}(undef, solution_size(base_problem)) for base_problem ∈ base_problems(problem)
        ]),
        mortar([
            Vector{T}(undef, solution_size(base_problem)) for base_problem ∈ base_problems(problem)
        ]),
        mortar([
            Vector{T}(undef, solution_size(base_problem)) for base_problem ∈ base_problems(problem)
        ]),
    )
end

function update_constraints!(problem, solver_cache::LoadDrivenSolverCache, t)
    Ferrite.update!(getch(problem), t)
    apply!(solver_cache.uₙ, getch(problem))
end

function update_constraints!(problem::CoupledProblem, solver_cache::LoadDrivenSolverCache, t)
    for (i,p) ∈ enumerate(problem.base_problems)
        update_constraints_block!(p, Block(i), solver_cache, t)
    end
end

function update_constraints_block!(problem, i::Block, solver_cache, t)
    Ferrite.update!(getch(problem), t)
    u = @view solver_cache.uₙ[i]
    apply!(u, getch(problem))
end

update_constraints_block!(problem::NullProblem, i::Block, solver_cache, t) = nothing

function perform_step!(problem, solver_cache::LoadDrivenSolverCache, t, Δt)
    solver_cache.uₙ₋₁ .= solver_cache.uₙ
    @info t
    update_constraints!(problem, solver_cache, t)
    if !solve!(solver_cache.uₙ, problem, solver_cache.inner_solver_cache, t) # TODO remove ,,t'' here. But how?
        @warn "Inner solver failed."
        return false
    end

    return true
end

function DiffEqBase.__init(
    prob::Thunderbolt.QuasiStaticProblem,
    alg::LoadDrivenSolver,
    args...;
    dt,
    tstops = (),
    saveat = nothing,
    save_everystep = false,
    callback = nothing,
    advance_to_tstop = false,
    save_func = (u, t) -> copy(u), # custom kwarg
    dtchangeable = true,           # custom kwarg
    stepstop = -1,                 # custom kwarg
    syncronizer = OS.NoExternalSynchronization(),
    kwargs...,
)
    (; u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    dt = tf > t0 ? dt : -dt

    _tstops = tstops
    _saveat = saveat
    tstops, saveat = OS.tstops_and_saveat_heaps(t0, tf, tstops, saveat)

    sol = DiffEqBase.build_solution(prob, alg, typeof(t0)[], typeof(save_func(u0, t0))[])

    callback = DiffEqBase.CallbackSet(callback)

    cache = setup_solver_cache(prob, alg, t0)

    cache.uₙ .= u0
    cache.uₙ₋₁ .= u0

    integrator = ThunderboltIntegrator(
        prob.f,
        cache.uₙ,
        nothing,
        cache.uₙ₋₁,
        1:length(u0),
        p,
        t0,
        t0,
        dt,
        cache,
        sol,
        true,
    )
    # DiffEqBase.initialize!(callback, u0, t0, integrator) # Do I need this?
    return integrator
end
