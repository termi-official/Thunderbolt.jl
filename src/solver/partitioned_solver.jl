abstract type AbstractPointwiseSolver <: AbstractSolver end
abstract type AbstractPointwiseSolverCache <: AbstractTimeSolverCache end

# Redirect to inner solve
function perform_step!(f::PointwiseODEFunction, cache::AbstractPointwiseSolverCache, t::Float64, Δt::Float64)
    perform_step!(f.ode, t, Δt, cache)
end

struct ForwardEulerCellSolver <: AbstractPointwiseSolver
end

mutable struct ForwardEulerCellSolverCache{uType, duType} <: AbstractPointwiseSolverCache
    du::duType
    uₙ::uType
    uₙ₋₁::uType
end

function perform_step!(cell_model::AbstractIonicModel, t::Float64, Δt::Float64, solver_cache::ForwardEulerCellSolverCache)
    # Eval buffer
    @unpack du, uₙ = solver_cache

    ndofs_local = 1+num_states(cell_model)
    npoints = length(uₙ)÷ndofs_local
    u_matrix = reshape(uₙ, (npoints,ndofs_local))

    # TODO formulate as a kernel for GPU
    for i ∈ 1:npoints
        u_local = @view u_matrix[i, :]
        # TODO this should happen in rhs call below
        @inbounds φₘ_cell = u_local[1]
        @inbounds s_cell  = @view u_local[2:end]

        # #TODO get x and Cₘ
        cell_rhs!(du, φₘ_cell, s_cell, nothing, t, cell_model)

        @inbounds u_local[1] = φₘ_cell + Δt*du[1]

        # # Non-allocating assignment
        @inbounds for j ∈ 1:num_states(cell_model)
            u_local[1+j] = s_cell[j] + Δt*du[j+1]
        end
    end

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::ForwardEulerCellSolver, t₀)
    @unpack npoints = f
    return ForwardEulerCellSolverCache(
        zeros(1+num_states(f.ode)),
        zeros(npoints*(num_states(f.ode)+1)),
        zeros(npoints*(num_states(f.ode)+1)),
    )
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
