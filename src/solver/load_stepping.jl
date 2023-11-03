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

# TODO revisiv if t₀ is really the right thing here to pass
function setup_solver_caches(problem::QuasiStaticNonlinearProblem, solver::LoadDrivenSolver{IS}, t₀) where {IS}
    inner_solver_cache = setup_solver_caches(problem, solver.inner_solver, t₀)
    @unpack dh = problem
    LoadDrivenSolverCache(
        inner_solver_cache,
        Vector{Float64}(undef, ndofs(dh)),
        Vector{Float64}(undef, ndofs(dh)),
    )
end

function setup_solver_caches(coupled_problem::CoupledProblem{<:Tuple{<:QuasiStaticNonlinearProblem,<:NullProblem}}, solver::LoadDrivenSolver{IS}, t₀) where {IS}
    problem = coupled_problem.base_problems[1]
    inner_solver_cache = setup_solver_caches(problem, solver.inner_solver, t₀)
    @unpack dh = problem
    LoadDrivenSolverCache(
        inner_solver_cache,
        Vector{Float64}(undef, ndofs(dh)),
        Vector{Float64}(undef, ndofs(dh)),
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

    if !solve!(solver_cache.uₜ, problem, solver_cache.inner_solver_cache, t) # TODO remove ,,t'' here. But how?
        @warn "Inner solver failed."
        return false
    end

    return true
end
