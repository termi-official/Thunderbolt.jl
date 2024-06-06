"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
end

mutable struct NewtonRaphsonSolverCache{OpType, ResidualType, T} <: AbstractNonlinearSolverCache
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonRaphsonSolver{T}
    #linear_solver_cache
end

function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::NewtonRaphsonSolver{T}) where {T}
    NewtonRaphsonSolverCache(setup_operator(f, solver), Vector{T}(undef, solution_size(f)), solver)
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::NewtonRaphsonSolver{T}) where {T}
    residual_buffer = mortar([
        Vector{T}(undef, bsize) for bsize ∈ blocksizes(f)
    ])
    NewtonRaphsonSolverCache(setup_operator(f, solver), residual_buffer, solver)
end

function nlsolve!(u::AbstractVector, f::AbstractSemidiscreteFunction, cache::NewtonRaphsonSolverCache, t)
    @unpack op, residual = cache
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, u, residual, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
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

        @timeit_debug "solve" !solve_inner_linear_system!(Δu, cache) && return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u .-= Δu # Current guess
    end
    return true
end
