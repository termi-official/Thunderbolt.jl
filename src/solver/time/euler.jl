#########################################################
#  This file contains optimized backward Euler solvers  #
#########################################################
Base.@kwdef struct BackwardEulerSolver{SolverType, SolutionVectorType, SystemMatrixType} <: AbstractSolver
    inner_solver::SolverType                       = LinearSolve.KrylovJL_CG()
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    system_matrix_type::Type{SystemMatrixType}     = ThreadedSparseMatrixCSR{Float64, Int64}
    # mass operator info
    # diffusion opeartor info
    verbose                                        = true # Temporary helper for benchmarks
end

struct BackwardEulerSolverCache{T, SolutionType <: AbstractVector{T}, StageType} <: AbstractTimeSolverCache
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # Temporary buffer for interpolations and stuff
    tmp::SolutionType
    # Utility to decide what kind of stage we solve (i.e. linear problem, full DAE or mass-matrix ODE)
    stage_info::StageType
end

# Performs a backward Euler step
function perform_step!(f, cache::BackwardEulerSolverCache, t, Δt)
    perform_backward_euler_step!(f, cache, cache.stage_info, t, Δt)
end

#########################################################
#                   Affine Problems                     #
#########################################################
# Mutable to change Δt_last
mutable struct BackwardEulerAffineStageInfo{T, MassMatrixType, DiffusionMatrixType, SourceTermType, SolverCacheType}
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    K::DiffusionMatrixType
    # Helper for possible source terms
    source_term::SourceTermType
    # Linear solver for (M - Δtₙ₋₁ K) uₙ = M uₙ₋₁ 
    linear_solver_cache::SolverCacheType
    # Last time step length as a check if we have to update K
    Δt_last::T
    # DO NOT USE THIS (will be replaced by proper logging system)
    verbose::Bool
end

function perform_backward_euler_step!(f, cache::BackwardEulerSolverCache, stage_info::BackwardEulerAffineStageInfo, t, Δt)
    @unpack uₙ, uₙ₋₁ = cache
    @unpack linear_solver_cache, M, Δt_last = stage_info

    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(stage_info, Δt)

    # Prepare right hand side b = M uₙ₋₁
    @timeit_debug "b = M uₙ₋₁" mul!(linear_solver_cache.b, M, uₙ₋₁)

    # Update source term
    @timeit_debug "update source term" begin
        implicit_euler_heat_update_source_term!(stage_info, t + Δt)
        add!(linear_solver_cache.b, stage_info.source_term)
    end

    # Solve linear problem, where sol.u === uₙ
    @timeit_debug "inner solve" sol = LinearSolve.solve!(linear_solver_cache)
    solve_failed = !(DiffEqBase.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == DiffEqBase.ReturnCode.Default)
    if stage_info.verbose || solve_failed # The latter seems off...
        if linear_solver_cache.cacheval !== nothing
            @info linear_solver_cache.cacheval.stats
        end
    end
    return !solve_failed
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerAffineStageInfo, Δt)
    _implicit_euler_heat_solver_update_system_matrix!(cache.linear_solver_cache.A, cache.M, cache.K, Δt)

    cache.Δt_last = Δt
end

function _implicit_euler_heat_solver_update_system_matrix!(A, M, K, Δt)
    # nonzeros(A) .= nonzeros(M.A) - Δt*nonzeros(K.A)
    nonzeros(A) .= nonzeros(K.A)
    nonzeros(A) .*= -Δt
    nonzeros(A) .+= nonzeros(M.A)
end

function implicit_euler_heat_update_source_term!(cache::BackwardEulerAffineStageInfo, t)
    needs_update(cache.source_term, t) && update_operator!(cache.source_term, t)
end

function setup_solver_cache(f::TransientDiffusionFunction, solver::BackwardEulerSolver, t₀)
    @unpack dh = f
    @unpack inner_solver = solver
    @assert length(dh.field_names) == 1 # TODO relax this assumption
    field_name = dh.field_names[1]

    A     = create_system_matrix(solver.system_matrix_type  , f)
    b     = create_system_vector(solver.solution_vector_type, f)
    u0    = create_system_vector(solver.solution_vector_type, f)
    uprev = create_system_vector(solver.solution_vector_type, f)
    tmp   = create_system_vector(solver.solution_vector_type, f)

    T = eltype(A)

    qr = create_quadrature_rule(f, solver, field_name)

    # Left hand side ∫dₜu δu dV
    mass_operator = setup_operator(
        BilinearMassIntegrator(
            ConstantCoefficient(T(1.0))
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

    inner_prob  = LinearSolve.LinearProblem(
        A, b; u0
    )
    inner_cache = init(inner_prob, inner_solver)

    cache       = BackwardEulerSolverCache(
        u0, # u
        uprev,
        tmp,
        BackwardEulerAffineStageInfo(
            mass_operator,
            diffusion_operator,
            source_operator,
            inner_cache,
            T(0.0),
            solver.verbose,
        )
    )

    @timeit_debug "initial assembly" begin
        update_operator!(mass_operator, t₀)
        update_operator!(diffusion_operator, t₀)
        update_operator!(source_operator, t₀)
    end

    return cache
end

#########################################################
#                     DAE Problems                     #
#########################################################

struct BackwardEulerDAEStageInfo{SolverType}
    nlsolver::SolverType
end

function perform_backward_euler_step!(f, cache::BackwardEulerSolverCache, stage_info::BackwardEulerDAEStageInfo, t, Δt)
    # What to do here?
    @warn "Not implemented yet."
    return false
end


# Multi-rate version
Base.@kwdef struct ForwardEulerSolver{SolutionVectorType} <: AbstractSolver
    rate::Int
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
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
        create_system_vector(solver.solution_vector_type, f),
        create_system_vector(solver.solution_vector_type, f),
        create_system_vector(solver.solution_vector_type, f),
        f.f
    )
end
