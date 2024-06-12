#########################################################
########################## TIME #########################
#########################################################
Base.@kwdef struct BackwardEulerSolver{SolverType, SolutionVectorType, SystemMatrixType} <: AbstractSolver
    inner_solver::SolverType = LinearSolve.KrylovJL_CG()
    solution_vector_type::Type{SolutionVectorType} = Vector
    system_matrix_type::Type{SystemMatrixType} = ThreadedSparseMatrixCSR
    # mass operator info
    # diffusion opeartor info
end

# TODO decouple from heat problem via special ODEFunction (AffineODEFunction)
mutable struct BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SourceTermType, SolverCacheType} <: AbstractTimeSolverCache
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    K::DiffusionMatrixType
    # Helper for possible source terms
    source_term::SourceTermType
    # Linear solver for (M - Δtₙ₋₁ K) uₙ = M uₙ₋₁
    inner_solver::SolverCacheType
    # Last time step length as a check if we have to update K
    Δt_last::Float64
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache, Δt)
    _implicit_euler_heat_solver_update_system_matrix!(cache.inner_solver.A, cache.M, cache.K, Δt)

    cache.Δt_last = Δt
end

_implicit_euler_heat_solver_update_system_matrix!(A, M, K, Δt) = @. A.nzval = M.A.nzval - Δt*K.A.nzval
_implicit_euler_heat_solver_update_system_matrix!(A::ThreadedSparseMatrixCSR, M, K, Δt) = _implicit_euler_heat_solver_update_system_matrix!(A.A, M, K, Δt)

function implicit_euler_heat_update_source_term!(cache::BackwardEulerSolverCache, t)
    needs_update(cache.source_term, t) && update_operator!(cache.source_term, t)
end

# Performs a backward Euler step
function perform_step!(f::TransientHeatFunction, cache::BackwardEulerSolverCache, t, Δt)
    @unpack Δt_last, M, uₙ, uₙ₋₁, inner_solver = cache
    # Remember last solution
    @inbounds uₙ₋₁ .= uₙ
    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(cache, Δt)
    # Prepare right hand side b = M uₙ₋₁
    @timeit_debug "b = M uₙ₋₁" mul!(inner_solver.b, M, uₙ₋₁)
    # TODO How to remove these two lines here?
    # Update source term
    @timeit_debug "update source term" begin
        implicit_euler_heat_update_source_term!(cache, t)
        add!(inner_solver.b, cache.source_term)
    end
    # Solve linear problem
    @timeit_debug "inner solve" LinearSolve.solve!(inner_solver)
    # @info inner_solver.cacheval.stats
    return true
end

function setup_solver_cache(f::TransientHeatFunction, solver::BackwardEulerSolver, t₀)
    @unpack dh = f
    @unpack inner_solver = solver
    @assert length(dh.field_names) == 1 # TODO relax this assumption, maybe.
    field_name = dh.field_names[1]

    qr = create_quadrature_rule(f, solver, field_name)

    # Left hand side ∫dₜu δu dV
    mass_operator = setup_operator(
        BilinearMassIntegrator(
            ConstantCoefficient(1.0)
        ),
        solver, dh, field_name, qr
    )

    # Affine right hand side ∫D grad(u) grad(δu) dV + ...
    diffusion_operator = setup_operator(
        BilinearDiffusionIntegrator(
            f.diffusion_tensor_field,
        ),
        solver, dh, field_name, qr
    )
    # ... + ∫f δu dV
    source_operator    = setup_operator(
        f.source_term,
        solver, dh, field_name, qr
    )

    A     = create_system_matrix(solver.system_matrix_type, f)
    b     = create_system_vector(solver.solution_vector_type, f)
    u0    = create_system_vector(solver.solution_vector_type, f)
    uprev = create_system_vector(solver.solution_vector_type, f)

    inner_prob  = LinearSolve.LinearProblem(
        A, b; u0
    )
    inner_cache = init(inner_prob, inner_solver)

    cache       = BackwardEulerSolverCache(
        u0, # u
        uprev,
        mass_operator,
        diffusion_operator,
        source_operator,
        inner_cache,
        0.0
    )

    @timeit_debug "initial assembly" begin
        update_operator!(mass_operator, t₀)
        update_operator!(diffusion_operator, t₀)
        update_operator!(source_operator, t₀)
    end

    return cache
end

# Multi-rate version
struct ForwardEulerSolver <: AbstractSolver
    rate::Int
end

mutable struct ForwardEulerSolverCache{VT,F} <: AbstractTimeSolverCache
    rate::Int
    du::VT
    uₙ::VT
    uₙ₋₁::VT
    rhs!::F
end

function perform_step!(f::ODEFunction, solver_cache::ForwardEulerSolverCache, t::Float64, Δt::Float64)
    @unpack rate, du, uₙ, rhs! = solver_cache
    Δtsub = Δt/rate
    for i ∈ 1:rate
        @inbounds rhs!(du, uₙ, t, f.p)
        @inbounds uₙ .= uₙ .+ Δtsub .* du
        t += Δtsub
    end

    return !any(isnan.(uₙ))
end

function setup_solver_cache(f::ODEFunction, solver::ForwardEulerSolver, t₀)
    return ForwardEulerSolverCache(
        solver.rate,
        zeros(num_states(f.ode)),
        zeros(num_states(f.ode)),
        zeros(num_states(f.ode)),
        f.f
    )
end
