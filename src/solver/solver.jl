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
    # TODO operators instead of directly storing the matrix!
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

adapt_timestep(t, Δt, problem, solver_cache) = (t += Δt, Δt)

"""
    solve(problem, solver, Δt, time_span, initial_condition, [callback])

Main entry point for solvers in Thunderbolt.jl. The design is inspired by
DifferentialEquations.jl. We try to upstream as much content as possible to
make it available for packages.
"""
function solve(problem, solver, Δt₀, (t₀, T), initial_condition, callback::CALLBACK = (t,p,c) -> nothing) where {CALLBACK}
    solver_cache = setup_solver_caches(problem, solver)

    setup_initial_condition!(problem, solver_cache, initial_condition)

    Δt = Δt₀
    t = t₀
    while t < T
        @info t, Δt
        if !perform_step!(problem, solver_cache, t, Δt)
            return false
        end

        callback(t, problem, solver_cache)

        t, Δt = adapt_timestep(t, Δt, problem, solver_cache)
    end

    @info T
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

# TODO what exactly is the job here? How do we know where to write and what to iterate?
function setup_initial_condition!(problem::SplitProblem{<:Any, <:AbstractPointwiseProblem}, cache, initial_condition)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    u₀, s₀ = initial_condition(problem)
    cache.B_solver_cache.uₙ .= u₀ # Note that the vectors in the caches are connected
    cache.B_solver_cache.sₙ .= s₀
    return nothing
end

perform_step!(problem::PointwiseODEProblem{ODET}, cache::CT, t::Float64, Δt::Float64) where {ODET, CT} = perform_step!(problem.ode, t, Δt, cache)



#########################################################
########################## TIME #########################
#########################################################
struct ImplicitEulerSolver
end

mutable struct ImplicitEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    K::DiffusionMatrixType
    # Buffer for (M - Δt K)
    A::SystemMatrixType
    # Linear solver for (M - Δtₙ₋₁ K) uₙ = M uₙ₋₁
    linsolver::LinSolverType
    # Buffer for right hand side
    b::RHSType
    # Last time step length as a check if we have to reassemble K
    Δt_last::Float64
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::ImplicitEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    cache.A = SystemMatrixType(cache.M.A - Δt*cache.K.A) # TODO FIXME make me generic
    cache.Δt_last = Δt
end

# Optimized version for CSR matrices
#TODO where is AbstractSparseMatrixCSR ?
# function implicit_euler_heat_solver_update_system_matrix!(cache::ImplicitEulerSolverCache{SolutionType, SMT1, SMT2, SMT2, LinSolverType, RHSType}, Δt) where {SolutionType, SMT1, SMT2, LinSolverType, RHSType}
#     cache.A = SparseMatrixCSR(transpose(cache.M.M - Δt*cache.K.K)) # TODO FIXME make me generic
# end

# Performs a backward Euler step
function perform_step!(problem, cache::ImplicitEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, t, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    @unpack Δt_last, b, M, A, uₙ, uₙ₋₁, linsolver = cache
    # Remember last solution
    @inbounds uₙ₋₁ .= uₙ
    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(cache, Δt)
    # Prepare right hand side b = M uₙ₋₁
    mul!(b, M, uₙ₋₁)
    # Solve linear problem
    # TODO abstraction layer and way to pass the solver/preconditioner pair (LinearSolve.jl?)
    Krylov.cg!(linsolver, A, b, uₙ₋₁)
    @inbounds uₙ .= linsolver.x

    return true
end

function setup_solver_caches(problem::TransientHeatProblem, solver::ImplicitEulerSolver)
    @unpack dh = problem
    @assert length(dh.field_names) == 1 # TODO relax this assumption, maybe.
    ip = dh.subdofhandlers[1].field_interpolations[1]
    order = Ferrite.getorder(ip)
    cv = CellValues(
        QuadratureRule{RefQuadrilateral}(2*order), # TODO how to pass this one down here?
        ip
    )
    cache = ImplicitEulerSolverCache(
        zeros(ndofs(dh)),
        zeros(ndofs(dh)),
        # TODO How to choose the exact operator types here?
        #      Maybe via some parameter in ImplicitEulerSolver?
        AssembledBilinearOperator(
            create_sparsity_pattern(dh),
            BilinearMassIntegrator(
                _->1.0, #ConstantCoefficient(1.0),
                cv
            ),
            cv,
            dh,
        ),
        AssembledBilinearOperator(
            create_sparsity_pattern(dh),
            BilinearDiffusionIntegrator(
                _->Tensor{2,2}((1.0, 0.0, 0.0, 1.0)), # problem.diffusion_tensor_field,
                cv,
            ),
            cv,
            dh,
        ),
        create_sparsity_pattern(dh),
        # TODO this via LinearSolvers.jl
        CgSolver(
            ndofs(dh),
            ndofs(dh),
            Vector{Float64}
        ),
        zeros(ndofs(dh)),
        0.0
    )

    # TODO where does this belong?
    update_operator!(cache.M)
    update_operator!(cache.K)

    return cache
end
