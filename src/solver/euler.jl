#########################################################
########################## TIME #########################
#########################################################
struct BackwardEulerSolver <: AbstractSolver
end

# TODO decouple from heat problem via special ODEFunction (AffineODEFunction)
mutable struct BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, SourceTermType, LinSolverType, RHSType}
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    J::DiffusionMatrixType
    # Buffer for (M - Δt J)
    A::SystemMatrixType
    # Helper for possible source terms
    source_term::SourceTermType
    # Linear solver for (M - Δtₙ₋₁ J) uₙ = M uₙ₋₁
    linsolver::LinSolverType
    # Buffer for right hand side
    b::RHSType
    # Last time step length as a check if we have to update J
    Δt_last::Float64
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache{<:Any, <:Any, <:Any, SystemMatrixType}, Δt) where {SystemMatrixType}
    cache.A = SystemMatrixType(cache.M.A - Δt*cache.J.A) # TODO FIXME make me generic
    cache.Δt_last = Δt
end
# Optimized version for CSR matrices - Lucky gamble dispatch
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache{<:Any, <:Any, <:Any, SystemMatrixType}, Δt) where {SystemMatrixType <: ThreadedSparseMatrixCSR}
    # We live in a symmeric utopia :)
    @timeit_debug "implicit_euler_heat_solver_update_system_matrix!" cache.A = SystemMatrixType(transpose(cache.M.A - Δt*cache.J.A)) # TODO FIXME make me generic
    cache.Δt_last = Δt
end

function implicit_euler_heat_update_source_term!(cache::BackwardEulerSolverCache, t)
    needs_update(cache.source_term, t) && update_operator!(cache.source_term, t)
end

# Performs a backward Euler step
# TODO check if operator is time dependent and update
function perform_step!(problem::TransientHeatProblem, cache::BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, t, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    @unpack Δt_last, b, M, A, uₙ, uₙ₋₁, linsolver = cache
    # Remember last solution
    @inbounds uₙ₋₁ .= uₙ
    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(cache, Δt)
    # Prepare right hand side b = M uₙ₋₁
    @timeit_debug "b = M uₙ₋₁" mul!(b, M, uₙ₋₁)
    # TODO How to remove these two lines here?
    # Update source term
    @timeit_debug "update source term" begin
        implicit_euler_heat_update_source_term!(cache, t)
        add!(b, cache.source_term)
    end
    # Solve linear problem
    # TODO abstraction layer and way to pass the solver/preconditioner pair (LinearSolve.jl?)
    @timeit_debug "inner solve" Krylov.cg!(linsolver, A, b, uₙ₋₁)
    @inbounds uₙ .= linsolver.x
    @info linsolver.stats
    return true
end

function setup_solver_cache(problem::TransientHeatProblem, solver::BackwardEulerSolver, t₀)
    @unpack dh = problem
    @assert length(dh.field_names) == 1 # TODO relax this assumption, maybe.
    field_name = dh.field_names[1]
    intorder = quadrature_order(problem, field_name)
    qr = QuadratureRuleCollection(intorder) # TODO how to pass this one down here?

    mass_operator = AssembledBilinearOperator(
        dh, field_name, # TODO field name
        BilinearMassIntegrator(
            ConstantCoefficient(1.0)
        ),
        qr
    )

    diffusion_operator = AssembledBilinearOperator(
        dh, field_name, # TODO field name
        BilinearDiffusionIntegrator(
            problem.diffusion_tensor_field,
        ),
        qr
    )

    cache = BackwardEulerSolverCache(
        zeros(solution_size(problem)),
        zeros(solution_size(problem)),
        # TODO How to choose the exact operator types here?
        #      Maybe via some parameter in BackwardEulerSolver?
        mass_operator,
        diffusion_operator,
        ThreadedSparseMatrixCSR(transpose(create_sparsity_pattern(dh))), # TODO this should be decided via some interface
        create_linear_operator(dh, problem.source_term),
        # TODO this via LinearSolvers.jl?
        CgSolver(
            solution_size(problem),
            solution_size(problem),
            Vector{Float64}
        ),
        zeros(solution_size(problem)),
        0.0
    )

    @timeit_debug "initial assembly" begin
        update_operator!(cache.M, t₀)
        update_operator!(cache.J, t₀)
    end
    
    return cache
end

# Multi-rate version
struct ForwardEulerSolver <: AbstractSolver
    rate::Int
end

mutable struct ForwardEulerSolverCache{VT,F}
    rate::Int
    du::VT
    uₙ::VT
    rhs!::F
end

function perform_step!(problem, solver_cache::ForwardEulerSolverCache, t::Float64, Δt::Float64)
    @unpack rate, du, uₙ, rhs! = solver_cache
    Δtsub = Δt/rate
    for i ∈ 1:rate
        @inbounds rhs!(du, uₙ, t, problem.p)
        @inbounds uₙ .= uₙ .+ Δtsub .* du
        t += Δtsub
    end

    return !any(isnan.(uₙ))
end

function setup_solver_cache(problem::ODEProblem, solver::ForwardEulerSolver, t₀)
    return ForwardEulerSolverCache(
        solver.rate,
        zeros(num_states(problem.ode)),
        zeros(num_states(problem.ode)),
        problem.f
    )
end
