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

mutable struct MultiLevelNewtonRaphsonSolverCache{OpType, ResidualType, T, InnerSolverCacheType} <: AbstractNonlinearSolverCache
    global_solver_cache::gCacheType
    local_solver_cache::lCacheType
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::NewtonRaphsonSolver{T}) where {T}
    MultiLevelNewtonRaphsonSolverCache(
        
    )
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    sizeu = solution_size(f)
    residual = Vector{T}(undef, sizeu)
    Δu = Vector{T}(undef, sizeu)
    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(
        getJ(op), residual; u0=Δu
    )
    inner_cache = init(inner_prob, inner_solver; alias_A=true, alias_b=true)
    @assert inner_cache.b === residual
    @assert inner_cache.A === getJ(op)

    NewtonRaphsonSolverCache(op, residual, solver, inner_cache)
end

function nlsolve!(u::AbstractVector, f::AbstractSemidiscreteFunction, cache::NewtonRaphsonSolverCache, t)
    @unpack op, residual, linear_solver_cache = cache
    newton_itr = -1
    Δu = linear_solver_cache.u
    while true
        newton_itr += 1

        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, residual, u, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix

        # vtk_grid("newton-debug-$newton_itr", problem.structural_problem.dh) do vtk
        #     vtk_point_data(vtk, f.structural_problem.dh, u[Block(1)])
        #     vtk_point_data(vtk, f.structural_problem.dh, residual[Block(1)], :residual)
        # end

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
