"""
    LoadDrivenSolver{IS, T, PFUN}

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
mutable struct LoadDrivenSolver{IS} <: AbstractSolver
    inner_solver::IS
end

mutable struct LoadDrivenSolverCache{ISC, T, VT <: AbstractVector{T}}
    inner_solver_cache::ISC
    uₙ::VT
    uₙ₋₁::VT
end

function setup_solver_cache(problem, solver::LoadDrivenSolver{<:NewtonRaphsonSolver{T}}, t₀) where T
    inner_solver_cache = setup_solver_cache(problem, solver.inner_solver)
    LoadDrivenSolverCache(
        inner_solver_cache,
        Vector{T}(undef, solution_size(problem)),
        Vector{T}(undef, solution_size(problem)),
    )
end

setup_solver_cache(problem::CoupledProblem, solver::LoadDrivenSolver, t₀) = error("Not implemented yet.")

function setup_solver_cache(problem::CoupledProblem, solver::LoadDrivenSolver{<:NewtonRaphsonSolver{T}}, t₀) where T
    inner_solver_cache = setup_solver_cache(problem, solver.inner_solver)
    LoadDrivenSolverCache(
        inner_solver_cache,
        mortar([
            Vector{T}(undef, solution_size(problem.base_problems[i])) for i ∈ 1:length(problem.base_problems)
        ]),
        mortar([
            Vector{T}(undef, solution_size(problem.base_problems[i])) for i ∈ 1:length(problem.base_problems)
        ]),
    )
end

function update_constraints!(problem, solver_cache::LoadDrivenSolverCache, t)
    Ferrite.update!(problem.ch, t)
    apply!(solver_cache.uₙ, problem.ch)
end

function update_constraints!(problem::CoupledProblem, solver_cache::LoadDrivenSolverCache, t)
    for (i,p) ∈ enumerate(problem.base_problems)
        update_constraints_block!(p, Block(i), solver_cache, t)
    end
end

function update_constraints_block!(problem, i::Block, solver_cache, t)
    Ferrite.update!(problem.ch, t)
    u = @view solver_cache.uₙ[i]
    apply!(u, problem.ch)
end

update_constraints_block!(problem::NullProblem, i::Block, solver_cache, t) = nothing

function perform_step!(problem, solver_cache::LoadDrivenSolverCache, t, Δt)
    solver_cache.uₙ₋₁ .= solver_cache.uₙ

    update_constraints!(problem, solver_cache, t)

    if !solve!(solver_cache.uₙ, problem, solver_cache.inner_solver_cache, t) # TODO remove ,,t'' here. But how?
        @warn "Inner solver failed."
        return false
    end

    return true
end
