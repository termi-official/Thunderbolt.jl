struct MultiLevelFunction{G,L}
    g::G # Global instance
    l::L # Local instance helper
end

"""
    MultilevelNewtonRaphsonSolver{T}

Multilevel Newton-Raphson solver [RabSanHsu:1979:mna](@ref) for nonlinear problems of the form `F(u,v) = 0; G(u,v) = 0`.
To use the Multilevel solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct MultiLevelNewtonRaphsonSolver{gSolverType <: NewtonRaphsonSolver, lSolverType <: NewtonRaphsonSolver} <: AbstractNonlinearSolver
    global_newton::gSolverType
    local_newton::lSolverType
end

struct MultiLevelNewtonRaphsonSolverCache{OpType, ResidualType, T, InnerSolverCacheType} <: AbstractNonlinearSolverCache
    global_solver_cache::gCacheType
    local_solver_cache::lCacheType
end

function setup_solver_cache(f, solver::MultiLevelNewtonRaphsonSolver{T}) where {T}
    MultiLevelNewtonRaphsonSolverCache(
        setup_solver_cache(f.g, solver.global_newton),
        setup_solver_cache(f.g, solver.local_newton),
    )
end

# This part should be the same as
function nlsolve!(u::AbstractVector, f::AbstractSemidiscreteFunction, cache::MultiLevelNewtonRaphsonSolver, t)
    @unpack op, residual, linear_solver_cache = cache
    newton_itr = -1
    Δu = linear_solver_cache.u
    while true
        newton_itr += 1

        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, residual, u, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix

        residualnorm = residual_norm(cache, f)
        @info "Newton itr $newton_itr: ||r||=$residualnorm"
        if residualnorm < cache.parameters.tol #|| (newton_itr > 0 && norm(Δu) < cache.parameters.tol)
            break
        elseif newton_itr > cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        elseif any(isnan.(residualnorm))
            @warn "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm"
            return false
        end

        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        @info "Linear solver stats: $(sol.stats) - norm(Δu) = $(norm(Δu))"
        solve_succeeded = LinearSolve.SciMLBase.successful_retcode(sol) || sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u .-= Δu # Current guess
    end
    return true
end
