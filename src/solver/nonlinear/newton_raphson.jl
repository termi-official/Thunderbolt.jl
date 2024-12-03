struct SummaryNewtonMonitor
end

@inline function newton_monitor_inner_callback(::SummaryNewtonMonitor, t, i, f, u, sol, linear_cache)
    @info "Linear solver stats: $(sol.stats) - norm(Δu) = $(norm(sol.u))"
end

struct VTKNewtonMonitor
    outdir::String
end

function newton_monitor_inner_callback(monitor::VTKNewtonMonitor, time, newton_itr, f, u, sol, linear_cache)
    @info "Linear solver stats: $(sol.stats) - norm(Δu) = $(norm(sol.u))"

    VTKGridFile(joinpath(monitor.outdir, "newton-monitor-t=$time-i=$newton_itr.vtu"), f.dh) do vtk
        write_solution(vtk, f.dh, u)
        write_solution(vtk, f.dh, linear_cache.b, "_residual")
        write_solution(vtk, f.dh, linear_cache.u, "_increment")
    end
end


"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T, solverType, M} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
    inner_solver::solverType = LinearSolve.KrylovJL_GMRES()
    monitor::M = SummaryNewtonMonitor()
end

mutable struct NewtonRaphsonSolverCache{OpType, ResidualType, T, NewtonType <: NewtonRaphsonSolver{T}, InnerSolverCacheType} <: AbstractNonlinearSolverCache
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonType
    linear_solver_cache::InnerSolverCacheType
    Θks::Vector{T} # TODO modularize this
    Δuprev::Vector{T}
end

function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    residual = Vector{T}(undef, solution_size(f))
    Δu = Vector{T}(undef, solution_size(f))

    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(
        getJ(op), residual; u0=Δu
    )
    inner_cache = init(inner_prob, inner_solver; alias_A=true, alias_b=true)
    @assert inner_cache.b === residual
    @assert inner_cache.A === getJ(op)

    NewtonRaphsonSolverCache(op, residual, solver, inner_cache, T[], copy(Δu))
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
    @unpack op, residual, linear_solver_cache, Θks, Δuprev = cache
    newton_itr = -1
    Δu = linear_solver_cache.u
    resize!(Θks, 0)
    while true
        newton_itr += 1
        Δuprev .= Δu
        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, residual, u, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix

        residualnorm = residual_norm(cache, f)
        @info "Newton itr $newton_itr: ||r||=$residualnorm"
        if residualnorm < cache.parameters.tol && newton_itr > 0 # Do at least two iterations to get a sane convergence estimate
            break
        elseif newton_itr > cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        elseif any(isnan.(residualnorm))
            @warn "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm"
            return false
        end

        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        newton_monitor_inner_callback(cache.parameters.monitor, t, newton_itr, f, u, sol, linear_solver_cache)
        solve_succeeded = LinearSolve.SciMLBase.successful_retcode(sol) || sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u .-= Δu # Current guess

        if newton_itr > 0
            Θk = norm(Δu)/norm(Δuprev)
            push!(Θks, Θk)
            if Θk ≥ 1.0
                @warn "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm"
                return false
            end
        end
    end
    return true
end
