"""
    update_linearization!(Jᵤ, residual, u, problem)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` and its residual `F(u)` in 
preparation to solve for the increment `Δu` with the linear problem `J(u) Δu = F(u)`.
"""
update_linearization!(Jᵤ, residual, u, problem) = error("Not overloaded")

"""
    update_linearization!(Jᵤ, u, problem)

Setup the linearized operator `Jᵤ(u)``.
"""
update_linearization!(Jᵤ, u, problem) = error("Not overloaded")

"""
    update_residual!(residual, u, problem)

Evaluate the residual `F(u)` of the problem.
"""
update_residual!(residual, u, problem) = error("Not overloaded")

"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T}
    # Convergence tolerance
    tol::T = 1e-8
    # Maximum number of iterations
    max_iter::Int = 100
end

mutable struct NewtonRaphsonSolverCache{JacType, ResidualType, T}
    # Cache for the Jacobian matrix df(u)/du
    J::JacType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonRaphsonSolver{T}
    #linear_solver_cache
end

function setup_solver_caches(problem, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack dh = problem
    # TODO operators instead of directly storing the matrix!
    NewtonRaphsonSolverCache(create_sparsity_pattern(dh), zeros(ndofs(dh)), solver)
end

eliminate_constraints_from_linearization!(solver_cache, problem) = apply_zero!(solver_cache.J, solver_cache.residual, problem.ch)

function solve!(u, problem, solver_cache::NewtonRaphsonSolverCache{JacType, ResidualType, T}) where {JacType, ResidualType, T}
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        update_linearization!(solver_cache.J, solver_cache.residual, u, problem)

        eliminate_constraints_from_linearization!(solver_cache, problem)

        residualnorm = norm(solver_cache.residual[Ferrite.free_dofs(problem.ch)])
        if residualnorm < solver_cache.parameters.tol
            break
        elseif newton_itr > solver_cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        end

        try
            solve_inner_linear_system!(Δu, solver_cache)
        catch err
            @warn "Linear solver failed: " , err
            return false
        end

        apply_zero!(Δu, problem.ch)

        u .-= Δu # Current guess
    end
    return true
end

function solve_inner_linear_system!(Δu, solver_cache::NewtonRaphsonSolverCache{JacType, ResidualType, T}) where {JacType, ResidualType, T}
    Δu .= solver_cache.J \ solver_cache.residual
end
