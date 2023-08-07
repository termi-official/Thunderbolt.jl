"""
    assemble_linearization!(Jᵤ, residual, u, problem_cache)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` and its residual `F(u)` in 
preparation to solve for the increment `Δu` with the linear problem `J(u) Δu = F(u)`.
"""
assemble_linearization!(Jᵤ, residual, u, problem_cache) = error("Not overloaded")

"""
    assemble_linearization!(Jᵤ, u, problem_cache)

Setup the linearized operator `Jᵤ(u)``.
"""
assemble_linearization!(Jᵤ, u, problem_cache) = error("Not overloaded")

"""
    assemble_residual!(residual, u, problem_cache)

Evaluate the residual `F(u)` of the problem.
"""
assemble_residual!(residual, u, problem_cache) = error("Not overloaded")

"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [assemble_linearization!](@ref)
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

function setup_solver_caches(dh, solver::NewtonRaphsonSolver{T}) where {T}
    NewtonRaphsonSolverCache(create_sparsity_pattern(dh), zeros(ndofs(dh)), solver)
end

function solve!(u, problem_cache, solver_cache::NewtonRaphsonSolverCache{JacType, ResidualType, T}) where {JacType, ResidualType, T}
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        assemble_linearization!(solver_cache.J, solver_cache.residual, u, problem_cache)

        rhsnorm = norm(solver_cache.residual[Ferrite.free_dofs(problem_cache.ch)])
        apply_zero!(solver_cache.J, solver_cache.residual, problem_cache.ch)

        if rhsnorm < solver_cache.parameters.tol
            break
        elseif newton_itr > solver_cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting."
            return false
        end

        try
            solve_inner_linear_system!(Δu, problem_cache, solver_cache)
        catch err
            @warn "Linear solver failed" , err
            return false
        end

        apply_zero!(Δu, problem_cache.ch)

        u .-= Δu # Current guess
    end
    return true
end

function solve_inner_linear_system!(Δu, problem_cache, solver_cache::NewtonRaphsonSolverCache{JacType, ResidualType, T}) where {JacType, ResidualType, T}
    Δu .= solver_cache.J \ solver_cache.residual
end

"""
    LoadDrivenSolver{IS, T, PFUN}

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
mutable struct LoadDrivenSolver{IS, T, PFUN}
    inner_solver::IS
    Δt::T
    t_begin::T
    t_end::T
    u_prev::Vector{T}
    postproc::PFUN # Takes (problem, u, t) and postprocesses it (includes IO operations) 
end

struct LoadDrivenSolverCache{IS, ISC, T, PFUN}
    inner_solver_cache::ISC
    parameters::LoadDrivenSolver{IS, T, PFUN}
end

function setup_solver_caches(dh, solver::LoadDrivenSolver{IS, T, PFUN}) where {IS, T, PFUN}
    LoadDrivenSolverCache(setup_solver_caches(dh, solver.inner_solver), solver)
end

function solve!(u₀, problem_cache, solver_cache::LoadDrivenSolverCache{IS, ISC, T}) where {IS, ISC, T}
    @unpack t_end, Δt, u_prev, postproc = solver_cache.parameters
    uₜ   = u₀
    uₜ₋₁ = copy(u₀)
    for t ∈ 0.0:Δt:t_end
        @info t

        problem_cache.t = t
        # Store last solution
        uₜ₋₁ .= uₜ
        problem_cache.u_prev .= uₜ₋₁

        # Update with new boundary conditions (if available)
        Ferrite.update!(problem_cache.ch, t)
        apply!(uₜ, problem_cache.ch)

        if !solve!(uₜ, problem_cache, solver_cache.inner_solver_cache)
            @warn "Inner solver failed."
            return false
        end

        postproc(problem_cache, uₜ, t)
    end

    return true
end
