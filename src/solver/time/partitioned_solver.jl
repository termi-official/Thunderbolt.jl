abstract type AbstractPointwiseSolver <: AbstractSolver end
abstract type AbstractPointwiseSolverCache <: AbstractTimeSolverCache end

# Redirect to inner solve
function perform_step!(f::PointwiseODEFunction, cache::AbstractPointwiseSolverCache, t::Real, Δt::Real)
    _pointwise_step_outer_kernel!(f, t, Δt, cache, cache.uₙ)
end

# This controls the outer loop over the ODEs
function _pointwise_step_outer_kernel!(f::PointwiseODEFunction, t::Real, Δt::Real, cache::AbstractPointwiseSolverCache, ::Vector)
    @unpack npoints = f

    @batch minbatch=cache.batch_size_hint for i ∈ 1:npoints
        _pointwise_step_inner_kernel!(f, i, t, Δt, cache) || return false
    end

    return true
end

"""
Simple forward euler to solve the cell model.
"""
Base.@kwdef struct ForwardEulerCellSolver{SolutionVectorType} <: AbstractPointwiseSolver
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    batch_size_hint::Int                           = 32
end

struct ForwardEulerCellSolverCache{duType, uType, dumType, umType} <: AbstractPointwiseSolverCache
    du::duType
    # These vectors hold the data
    uₙ::uType
    uₙ₋₁::uType
    tmp::uType
    # These array view the data above to give easy indices of the form [ode index, local state index]
    dumat::dumType
    uₙmat::umType
    # uₙ₋₁mat::umType
    batch_size_hint::Int
end

# This is the actual solver
function _pointwise_step_inner_kernel!(f::PointwiseODEFunction, i::Int, t::Real, Δt::Real, cache::ForwardEulerCellSolverCache)
    cell_model = f.ode
    u_local    = @view cache.uₙmat[i, :]
    du_local   = @view cache.dumat[i, :]
    # TODO this should happen in rhs call below
    @inbounds φₘ_cell = u_local[1]
    @inbounds s_cell  = @view u_local[2:end]

    # #TODO get spatial coordinate x and Cₘ
    cell_rhs!(du_local, φₘ_cell, s_cell, nothing, t, cell_model)

    @. u_local = u_local + Δt*du_local

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::ForwardEulerCellSolver, t₀)
    @unpack npoints, ode = f
    ndofs_local = num_states(ode)

    du      = create_system_vector(solver.solution_vector_type, f)
    dumat   = reshape(du, (npoints,ndofs_local))
    uₙ      = create_system_vector(solver.solution_vector_type, f)
    uₙ₋₁    = create_system_vector(solver.solution_vector_type, f)
    tmp     = create_system_vector(solver.solution_vector_type, f)
    uₙmat   = reshape(uₙ, (npoints,ndofs_local))

    return ForwardEulerCellSolverCache(du, uₙ, uₙ₋₁, tmp, dumat, uₙmat, solver.batch_size_hint)
end

Base.@kwdef struct AdaptiveForwardEulerSubstepper{T, SolutionVectorType <: AbstractVector{T}} <: AbstractPointwiseSolver
    substeps::Int                                  = 10
    reaction_threshold::T                          = 0.1
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    batch_size_hint::Int                           = 32
end

struct AdaptiveForwardEulerSubstepperCache{T, duType, uType, dumType, umType} <: AbstractPointwiseSolverCache
    du::duType
    # These vectors hold the data
    uₙ::uType
    uₙ₋₁::uType
    # These array view the data above to give easy indices of the form [ode index, local state index]
    dumat::dumType
    uₙmat::umType
    # Solver parameters
    substeps::Int
    reaction_threshold::T
    batch_size_hint::Int
end

function _pointwise_step_inner_kernel!(f::PointwiseODEFunction, i::Int, t::Real, Δt::Real, cache::AdaptiveForwardEulerSubstepperCache)
    cell_model = f.ode
    u_local    = @view cache.uₙmat[i, :]
    du_local   = @view cache.dumat[i, :]
    # TODO this should happen in rhs call below
    @inbounds φₘ_cell = u_local[1]
    @inbounds s_cell  = @view u_local[2:end]

    # #TODO get spatial coordinate x and Cₘ
    cell_rhs!(du_local, φₘ_cell, s_cell, nothing, t, cell_model)

    if du_local[1] < cache.reaction_threshold
        @. u_local = u_local + Δt*du_local
    else
        Δtₛ = Δt/cache.substeps
        @. u_local = u_local + Δtₛ*du_local

        for substep ∈ 2:cache.substeps
            tₛ = t + substep*Δtₛ
            @inbounds φₘ_cell = u_local[1]

            #TODO get x and Cₘ
            cell_rhs!(du_local, φₘ_cell, s_cell, nothing, tₛ, cell_model)

            @. u_local = u_local + Δtₛ*du_local
        end
    end

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::AdaptiveForwardEulerSubstepper, t₀)
    @unpack npoints, ode = f
    ndofs_local = num_states(ode)

    du      = create_system_vector(solver.solution_vector_type, f)
    dumat   = reshape(du, (npoints,ndofs_local))
    uₙ      = create_system_vector(solver.solution_vector_type, f)
    uₙ₋₁    = create_system_vector(solver.solution_vector_type, f)
    uₙmat   = reshape(uₙ, (npoints,ndofs_local))

    return AdaptiveForwardEulerSubstepperCache(du, uₙ, uₙ₋₁, dumat, uₙmat, solver.substeps, solver.reaction_threshold, solver.batch_size_hint)
end
