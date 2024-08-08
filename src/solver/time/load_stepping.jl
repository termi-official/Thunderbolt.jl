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

function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::LoadDrivenSolver, t₀)
    inner_solver_cache = setup_solver_cache(f, solver.inner_solver)
    T = Float64 # TODO query
    vtype = Vector{T}
    LoadDrivenSolverCache(
        inner_solver_cache,
        vtype(undef, solution_size(f)),
        vtype(undef, solution_size(f)),
        vtype(undef, solution_size(f)),
    )
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::LoadDrivenSolver, t₀)
    inner_solver_cache = setup_solver_cache(f, solver.inner_solver)
    T = Float64 # TODO query
    vtype = Vector{T}
    LoadDrivenSolverCache(
        inner_solver_cache,
        mortar([
            vtype(undef, solution_size(fi)) for fi ∈ blocks(f)
        ]),
        mortar([
            vtype(undef, solution_size(fi)) for fi ∈ blocks(f)
        ]),
        mortar([
            vtype(undef, solution_size(fi)) for fi ∈ blocks(f)
        ]),
    )
end

function perform_step!(f::AbstractSemidiscreteFunction, solver_cache::LoadDrivenSolverCache, t, Δt)
    solver_cache.uₙ₋₁ .= solver_cache.uₙ
    @info t
    update_constraints!(f, solver_cache, t + Δt)
    if !nlsolve!(solver_cache.uₙ, f, solver_cache.inner_solver_cache, t + Δt) # TODO remove ,,t'' here. But how?
        @warn "Inner solver failed."
        return false
    end

    return true
end
