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
#     inner_solver::SolverType
#     cells_per_thread::Int = 64
# end

# struct ThreadedCellSolverCache{CacheType<:AbstractPointwiseSolverCache} <: AbstractPointwiseSolverCache
#     scratch::Vector{CacheType}
# end

# function setup_solver_caches(problem, solver::ThreadedCellSolver{InnerSolverType}, t₀) where {InnerSolverType}
#     @unpack npoints = problem # TODO what is a good abstraction layer over this?
#     return ThreadedCellSolverCache(
#         [setup_solver_caches(problem, solver.inner_solver, t₀) for i ∈ 1:solver.cells_per_thread]
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

# # TODO better abstraction layer
# function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{BackwardEulerSolver,BST}, t₀) where {APT <: TransientHeatProblem,BPT,BST}
#     cache = LTGOSSolverCache(
#         setup_solver_caches(problem.A, solver.A_solver, t₀),
#         setup_solver_caches(problem.B, solver.B_solver, t₀),
#     )
#     cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
#     return cache
# end

# function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{AST,BackwardEulerSolver}, t₀) where {APT,BPT <: TransientHeatProblem,AST}
#     cache = LTGOSSolverCache(
#         setup_solver_caches(problem.A, solver.A_solver, t₀),
#         setup_solver_caches(problem.B, solver.B_solver, t₀),
#     )
#     cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
#     return cache
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

# TODO decouple the concept ForwardEuler from "CellProblem"
function setup_solver_caches(problem, solver::ThreadedForwardEulerCellSolver, t₀)
    @unpack npoints = problem # TODO what is a good abstraction layer over this?
    return ThreadedForwardEulerCellSolverCache(
        zeros(1+num_states(problem.ode)),
        zeros(npoints),
        zeros(npoints, num_states(problem.ode)),
        solver.num_cells_per_batch
    )
end

struct RushLarsenSolver
end

struct RushLarsenSolverCache
end

function setup_solver_caches(problem::PartitionedProblem{APT, BPT}, solver::RushLarsenSolver) where {APT,BPT}
    return RushLarsenSolverCache()
end
