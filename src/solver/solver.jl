"""
    assemble_linearization!(Jᵤ, residual, u, problem)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` and its residual `F(u)` in 
preparation to solve for the increment `Δu` with the linear problem `J(u) Δu = F(u)`.
"""
assemble_linearization!(Jᵤ, residual, u, problem) = error("Not overloaded")

"""
    assemble_linearization!(Jᵤ, u, problem)

Setup the linearized operator `Jᵤ(u)``.
"""
assemble_linearization!(Jᵤ, u, problem) = error("Not overloaded")

"""
    assemble_residual!(residual, u, problem)

Evaluate the residual `F(u)` of the problem.
"""
assemble_residual!(residual, u, problem) = error("Not overloaded")

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

function setup_solver_caches(problem, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack dh = problem
    NewtonRaphsonSolverCache(create_sparsity_pattern(dh), zeros(ndofs(dh)), solver)
end

eliminate_constraints_from_linearization!(solver_cache, problem) = apply_zero!(solver_cache.J, solver_cache.residual, problem.ch)

function solve!(u, problem, solver_cache::NewtonRaphsonSolverCache{JacType, ResidualType, T}) where {JacType, ResidualType, T}
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        assemble_linearization!(solver_cache.J, solver_cache.residual, u, problem)

        eliminate_constraints_from_linearization!(solver_cache, problem)

        residualnorm = norm(solver_cache.residual[Ferrite.free_dofs(problem.ch)])
        @show residualnorm
        if residualnorm < solver_cache.parameters.tol
            break
        elseif newton_itr > solver_cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting."
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

function setup_solver_caches(problem, solver::LoadDrivenSolver{IS}) where {IS}
    LoadDrivenSolverCache(
        setup_solver_caches(problem, solver.inner_solver),
        zeros(ndofs(problem.dh)),
        zeros(ndofs(problem.dh)),
    )
end

function setup_initial_condition!(problem, solver_cache::LoadDrivenSolverCache, initial_condition)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    solver_cache.uₜ = zeros(ndofs(problem.dh))
    solver_cache.uₜ₋₁ = zeros(ndofs(problem.dh))
    return nothing
end

function perform_step!(problem, solver_cache::LoadDrivenSolverCache, t, Δt)
    # TODO remove these lines
    problem.t = t
    problem.Δt = Δt
    problem.u_prev .= solver_cache.uₜ₋₁

    solver_cache.uₜ₋₁ .= solver_cache.uₜ

    Ferrite.update!(problem.ch, t)
    apply!(solver_cache.uₜ, problem.ch)

    if !solve!(solver_cache.uₜ, problem, solver_cache.inner_solver_cache)
        @warn "Inner solver failed."
        return false
    end

    return true
end

function solve(problem, solver::LoadDrivenSolver, Δt₀, (t₀, T), initial_condition, callback::CALLBACK = (t,p,c) -> nothing) where {CALLBACK}
    solver_cache = setup_solver_caches(problem, solver)

    setup_initial_condition!(problem, solver_cache, initial_condition)

    Δt = Δt₀
    t = t₀
    while t < T
        @info t, Δt
        # @timeit "solver" begin 
            if !perform_step!(problem, solver_cache, t, Δt)
                #return false
            end
        # end

        callback(t, problem, solver_cache)

        # TODO Δt adaption
        t += Δt
    end

    @info T
    # @timeit "solver" perform_step!(problem, solver_cache, T, T-t)
    if !perform_step!(problem, solver_cache, t, T-t)
        return false
    end

    callback(t, problem, solver_cache)
    
    return true
end

# struct JacobiPreconditioner{T}
#     Ainv::Diagonal{T}
# end
# function JacobiPreconditioner(A)
#     JacobiPreconditioner{eltype(A)}(inv(Diagonal(diag(A))))
# end
# LinearAlgebra.ldiv!(P::JacobiPreconditioner{T}, b) where {T} = mul!(b, P.Ainv, b)
# LinearAlgebra.ldiv!(y, P::JacobiPreconditioner{T}, b) where {T} = mul!(y, P.Ainv, b)
# import Base: \
# function (\)(P::JacobiPreconditioner{T}, b) where {T}
#     return ldiv!(similar(b), P.Ainv, b)
# end

# using Polyester
# struct JacobiPreconditioner{T}
#     Ainv::Vector{T}
# end
# function JacobiPreconditioner(A)
#     JacobiPreconditioner{eltype(A)}(inv.(Vector(diag(A))))
# end
# function LinearAlgebra.ldiv!(y::TV, P::JacobiPreconditioner{T}, b::TV) where {T, TV<:AbstractVector}
#     @batch minbatch = size(y, 1) ÷ Threads.nthreads() for row in 1:length(P.Ainv)
#         @inbounds begin
#             y[row] = P.Ainv[row]*b[row]
#         end
#     end
# end
