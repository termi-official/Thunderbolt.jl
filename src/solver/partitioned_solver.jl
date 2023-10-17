abstract type AbstractPointwiseSolver end
abstract type AbstractPointwiseSolverCache end

struct ForwardEulerCellSolver <: AbstractPointwiseSolver
end

mutable struct ForwardEulerCellSolverCache{VT, MT} <: AbstractPointwiseSolverCache
    du::VT
    uₙ::VT
    sₙ::MT
end

function perform_step!(cell_model::ION, t::Float64, Δt::Float64, solver_cache::ForwardEulerCellSolverCache{VT}) where {VT, ION <: AbstractIonicModel}
    # Eval buffer
    @unpack du, uₙ, sₙ = solver_cache

    # TODO formulate as a kernel for GPU
    for i ∈ 1:length(uₙ)
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

    return true
end

# TODO decouple the concept ForwardEuler from "CellProblem"
function setup_solver_caches(problem, solver::ForwardEulerCellSolver, t₀)
    @unpack npoints = problem # TODO what is a good abstraction layer over this?
    return ForwardEulerCellSolverCache(
        zeros(1+num_states(problem.ode)),
        zeros(npoints),
        zeros(npoints, num_states(problem.ode))
    )
end

Base.@kwdef struct AdaptiveForwardEulerReactionSubCellSolver{T} <: AbstractPointwiseSolver
    substeps::Int = 10
    reaction_threshold::T = 0.1
end

struct AdaptiveForwardEulerReactionSubCellSolverCache{VT, T} <: AbstractPointwiseSolverCache
    uₙ::VT
    sₙ::VT
    du::VT
    substeps::Int
    reaction_threshold::T
end

function perform_step!(cell_model::ION, t::Float64, Δt::Float64, cache::AdaptiveForwardEulerReactionSubCellSolverCache{VT,T}) where {VT, T, ION <: AbstractIonicModel}
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
#     cells_per_thread::Int = 64
# end

# Base.@kwdef struct ThreadedCellSolverCache{CacheType<:AbstractPointwiseSolverCache} <: AbstractPointwiseSolverCache
#     scratch::Vector{CacheType}
#     cells_per_thread::Int = 64
# end

# function perform_step!(uₙ::T1, sₙ::T2, cell_model::ION, t::Float64, Δt::Float64, cache::ThreadedCellSolverCache{CacheType}) where {T1, T2, CacheType, ION <: AbstractIonicModel}
#     for tid ∈ 1:cache.cells_per_thread:length(uₙ)
#         tcache = cache.scratch[Threads.threadid()]
#         last_cell_in_thread = min((tid+cells_per_thread),length(uₙ))
#         tuₙ = @view uₙ[tid:last_cell_in_thread]
#         tsₙ = @view sₙ[tid:last_cell_in_thread]
#         Threads.@threads for tid ∈ 1:cells_per_thread:length(uₙ)
#             perform_step!(tuₙ, tsₙ, cell_model, t, Δt, tcache)
#         end
#     end
# end

struct RushLarsenSolver
end

struct RushLarsenSolverCache
end

function setup_solver_caches(problem::PartitionedProblem{APT, BPT}, solver::RushLarsenSolver) where {APT,BPT}
    return RushLarsenSolverCache()
end
