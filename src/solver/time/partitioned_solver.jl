abstract type AbstractPointwiseSolver <: AbstractSolver end
abstract type AbstractPointwiseSolverCache <: AbstractTimeSolverCache end

# Redirect to inner solve
function perform_step!(f::PointwiseODEFunction, cache::AbstractPointwiseSolverCache, t::Real, Δt::Real)
    _pointwise_step_outer_kernel!(f, t, Δt, cache, cache.uₙ)
end

Base.@kwdef struct ForwardEulerCellSolver{SolutionVectorType} <: AbstractPointwiseSolver
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    batch_size_hint::Int                           = 32
end

mutable struct ForwardEulerCellSolverCache{duType, uType, dumType, umType} <: AbstractPointwiseSolverCache
    du::duType
    # These vectors hold the data
    uₙ::uType
    uₙ₋₁::uType
    # These array view the data above to give easy indices of the form [ode index, local state index]
    dumat::dumType
    uₙmat::umType
    # uₙ₋₁mat::umType
end

# This controls the outer loop over the ODEs
function _pointwise_step_outer_kernel!(f::PointwiseODEFunction, t::Real, Δt::Real, solver_cache::AbstractPointwiseSolverCache, ::Vector)
    @unpack npoints = f

    @batch for i ∈ 1:npoints
        _pointwise_step_inner_kernel!(f, i, t, Δt, solver_cache) || return false
    end

    return true
end

# This is the actual solver
function _pointwise_step_inner_kernel!(f::PointwiseODEFunction, i::Int, t::Real, Δt::Real, solver_cache::ForwardEulerCellSolverCache)
    cell_model = f.ode
    u_local = @view solver_cache.uₙmat[i, :]
    du_local = @view solver_cache.dumat[i, :]
    # TODO this should happen in rhs call below
    @inbounds φₘ_cell = u_local[1]
    @inbounds s_cell  = @view u_local[2:end]

    # #TODO get spatial coordinate x and Cₘ
    cell_rhs!(du_local, φₘ_cell, s_cell, nothing, t, cell_model)

    @inbounds u_local[1] = φₘ_cell + Δt*du_local[1]

    # # Non-allocating assignment
    @inbounds for j ∈ 1:num_states(cell_model)
        u_local[1+j] = s_cell[j] + Δt*du_local[j+1]
    end

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::ForwardEulerCellSolver, t₀)
    @unpack npoints, ode = f
    ndofs_local = num_states(ode)

    du      = create_system_vector(solver.solution_vector_type, f)
    dumat   = reshape(du, (npoints,ndofs_local))
    uₙ      = create_system_vector(solver.solution_vector_type, f)
    uₙ₋₁    = create_system_vector(solver.solution_vector_type, f)
    uₙmat   = reshape(uₙ, (npoints,ndofs_local))
    # uₙ₋₁mat = reshape(uₙ₋₁, (npoints,ndofs_local))

    # return ForwardEulerCellSolverCache(du, uₙ, uₙ₋₁, uₙmat, uₙ₋₁mat)
    return ForwardEulerCellSolverCache(du, uₙ, uₙ₋₁, dumat, uₙmat)
end

Base.@kwdef struct AdaptiveForwardEulerReactionSubCellSolver{T} <: AbstractPointwiseSolver
    substeps::Int = 10
    reaction_threshold::T = 0.1
end

struct AdaptiveForwardEulerReactionSubCellSolverCache{T, VT <: AbstractVector{T}} <: AbstractPointwiseSolverCache
    uₙ::VT
    sₙ::VT
    du::VT
    substeps::Int
    reaction_threshold::T
end

function perform_step!(cell_model::AbstractIonicModel, t::Real, Δt::Real, cache::AdaptiveForwardEulerReactionSubCellSolverCache)
    @unpack uₙ, sₙ, du = cache

    # TODO formulate as a kernel for GPU
    for i ∈ 1:length(uₙ)
        @inbounds φₘ_cell = uₙ[i]
        @inbounds s_cell  = @view sₙ[i,:]

        #TODO get x and Cₘ
        cell_rhs!(du, φₘ_cell, s_cell, nothing, t, cell_model)

        if du[1] < cache.reaction_threshold
            @inbounds uₙ[i] = φₘ_cell + Δt*du[1]

            # Non-allocating assignment
            @inbounds for j ∈ 1:num_states(cell_model)
                sₙ[i,j] = s_cell[j] + Δt*du[j+1]
            end
        else
            Δtₛ = Δt/cache.substeps

            @inbounds uₙ[i] = φₘ_cell + Δtₛ*du[1]

            # Non-allocating assignment
            @inbounds for j ∈ 1:num_states(cell_model)
                sₙ[i,j] = s_cell[j] + Δtₛ*du[j+1]
            end

            for substep ∈ 2:solver.substeps
                tₛ = t + substep*Δtₛ

                @inbounds φₘ_cell = uₙ[i]
                @inbounds s_cell  = @view sₙ[i,:]

                #TODO get x and Cₘ
                cell_rhs!(du, φₘ_cell, s_cell, nothing, tₛ, cell_model)

                @inbounds uₙ[i] = φₘ_cell + Δtₛ*du[1]

                # Non-allocating assignment
                @inbounds for j ∈ 1:num_states(cell_model)
                    sₙ[i,j] = s_cell[j] + Δtₛ*du[j+1]
                end
            end
        end
    end

    return true
end

# Base.@kwdef struct ThreadedCellSolver{SolverType<:AbstractPointwiseSolver} <: AbstractPointwiseSolver
#     inner_solver::SolverType
#     cells_per_thread::Int = 64
# end

# struct ThreadedCellSolverCache{CacheType<:AbstractPointwiseSolverCache} <: AbstractPointwiseSolverCache
#     scratch::Vector{CacheType}
# end

# function setup_solver_cache(f, solver::ThreadedCellSolver{InnerSolverType}, t₀) where {InnerSolverType}
#     @unpack npoints = f # TODO what is a good abstraction layer over this?
#     return ThreadedCellSolverCache(
#         [setup_solver_cache(f, solver.inner_solver, t₀) for i ∈ 1:solver.cells_per_thread]
#     )
# end

# function perform_step!(uₙ::T1, sₙ::T2, cell_model::ION, t::Float64, Δt::Float64, cache::ThreadedCellSolverCache{CacheType}) where {T1, T2, CacheType, ION <: AbstractIonicModel}
#     for tid ∈ 1:cache.cells_per_thread:length(uₙ)
#         tcache = cache.scratch[Threads.threadid()]
#         last_cell_in_thread = min((tid+cells_per_thread),length(uₙ))
#         tuₙ = @view uₙ[tid:last_cell_in_thread]
#         tsₙ = @view sₙ[tid:last_cell_in_thread]
#         Threads.@threads :static for tid ∈ 1:cells_per_thread:length(uₙ)
#             perform_step!(tuₙ, tsₙ, cell_model, t, Δt, tcache)
#         end
#     end
# end

# TODO fix the one above somehow

struct ThreadedForwardEulerCellSolver <: AbstractPointwiseSolver
    num_cells_per_batch::Int
end

mutable struct ThreadedForwardEulerCellSolverCache{VT, MT} <: AbstractPointwiseSolverCache
    du::VT
    uₙ::VT
    sₙ::MT
    num_cells_per_batch::Int
end

function perform_step!(cell_model::ION, t::Float64, Δt::Float64, solver_cache::ThreadedForwardEulerCellSolverCache{VT}) where {VT, ION <: AbstractIonicModel}
    # Eval buffer
    @unpack du, uₙ, sₙ = solver_cache

    Threads.@threads :static for j ∈ 1:solver_cache.num_cells_per_batch:length(uₙ)
        for i ∈ j:min(j+solver_cache.num_cells_per_batch-1, length(uₙ))
            @inbounds φₘ_cell = uₙ[i]
            @inbounds s_cell  = @view sₙ[i,:]

            # #TODO get x and Cₘ
            cell_rhs!(du, φₘ_cell, s_cell, nothing, t, cell_model)

            @inbounds uₙ[i] = φₘ_cell + Δt*du[1]

            # # Non-allocating assignment
            @inbounds for j ∈ 1:num_states(cell_model)
                sₙ[i,j] = s_cell[j] + Δt*du[j+1]
            end
        end
    end

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::ThreadedForwardEulerCellSolver, t₀)
    @unpack npoints = f # TODO what is a good abstraction layer over this?
    return ThreadedForwardEulerCellSolverCache(
        zeros(1+num_states(f.ode)),
        zeros(npoints),
        zeros(npoints, num_states(f.ode)),
        solver.num_cells_per_batch
    )
end
