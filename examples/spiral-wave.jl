using Thunderbolt, LinearAlgebra, SparseArrays, UnPack
import Thunderbolt: AbstractIonicModel

using TimerOutputs, BenchmarkTools

using Krylov

using SparseMatricesCSR, ThreadedSparseCSR
ThreadedSparseCSR.multithread_matmul(PolyesterThreads())

######################################################
Base.@kwdef struct ParametrizedFHNModel{T} <: AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T = T(0.01)
end;

const FHNModel = ParametrizedFHNModel{Float64};

num_states(::ParametrizedFHNModel{T}) where{T} = 1
default_initial_state(::ParametrizedFHNModel{T}) where {T} = [0.0, 0.0]

function cell_rhs!(du::TD,φₘ::TV,s::TS,x::TX,t::TT,cell_parameters::TP) where {TD,TV,TS,TX,TT,TP <: AbstractIonicModel}
    dφₘ = @view du[1:1]
    reaction_rhs!(dφₘ,φₘ,s,x,t,cell_parameters)

    ds = @view du[2:end]
    state_rhs!(ds,φₘ,s,x,t,cell_parameters)

    return nothing
end

@inline function reaction_rhs!(dφₘ::TD,φₘ::TV,s::TS,x::TX,t::TT,cell_parameters::FHNModel) where {TD<:SubArray,TV,TS,TX,TT}
    @unpack a = cell_parameters
    dφₘ .= φₘ*(1-φₘ)*(φₘ-a) -s[1]
    return nothing
end

@inline function state_rhs!(ds::TD,φₘ::TV,s::TS,x::TX,t::TT,cell_parameters::FHNModel) where {TD<:SubArray,TV,TS,TX,TT}
    @unpack b,c,d,e = cell_parameters
    ds .= e*(b*φₘ - c*s[1] - d)
    return nothing
end

######################################################

epmodel = MonodomainModel(
    x->1.0,
    x->1.0,
    x->SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)),
    NoStimulationProtocol(),
    FHNModel()
)

# TODO where to put this setup?
grid = generate_grid(Quadrilateral, (256, 256), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))
# addnodeset!(grid, "ground", x-> x[2] == -0 && x[1] == -0)
dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :ϕₘ, 1)
# push!(dh, :ϕₑ, 1)
# push!(dh, :s, 1)
close!(dh);

# Initial condition
# TODO apply_analytical!
u₀ = zeros(ndofs(dh));
s₀ = zeros(ndofs(dh),num_states(epmodel.ion));
for cell in CellIterator(dh)
    _celldofs = celldofs(cell)
    ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
    for (i, coordinate) in enumerate(getcoordinates(cell))
        if coordinate[1] <= 1.25 && coordinate[2] <= 1.25
            u₀[ϕₘ_celldofs[i]] = 1.0
        end
        if coordinate[2] >= 1.25
            s₀[ϕₘ_celldofs[i],1] = 0.1
        end
    end
end

######################################################
struct ImplicitEulerHeatSolver end
mutable struct ImplicitEulerHeatSolverCache{MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    M::MassMatrixType
    K::DiffusionMatrixType
    A::SystemMatrixType
    linsolver::LinSolverType
    b::RHSType
end

function assemble_global!(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, M::SparseMatrixCSC, dh::DofHandler, model::MonodomainModel) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    Kₑ = zeros(n_basefuncs, n_basefuncs)
    Mₑ = zeros(n_basefuncs, n_basefuncs)

    assembler_K = start_assemble(K)
    assembler_M = start_assemble(M)

    @inbounds for cell in CellIterator(dh)
        fill!(Kₑ, 0)
        fill!(Mₑ, 0)
        #get the coordinates of the current cell
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            #get the spatial coordinates of the current gauss point
            x = spatial_coordinate(cellvalues, q_point, coords)
            #based on the gauss point coordinates, we get the spatial dependent
            #material parameters
            κ_loc = model.κ(x)
            Cₘ_loc = model.Cₘ(x)
            χ_loc = model.χ(x)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                Nᵢ = shape_value(cellvalues, q_point, i)
                ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    Nⱼ = shape_value(cellvalues, q_point, j)
                    ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                    Kₑ[i,j] -= ((κ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Mₑ[i,j] += Cₘ_loc * χ_loc * Nᵢ * Nⱼ * dΩ 
                end
            end
        end

        assemble!(assembler_K, celldofs(cell), Kₑ)
        assemble!(assembler_M, celldofs(cell), Mₑ)
    end
end

function solve!(uₙ, uₙ₋₁, t, Δt, cache::ImplicitEulerHeatSolverCache)
    mul!(cache.b, cache.M, uₙ₋₁)
    Krylov.cg!(cache.linsolver, cache.A, cache.b, uₙ₋₁)
    uₙ .= cache.linsolver.x
end

######################################################

abstract type AbstractCellSolver end
abstract type AbstractCellSolverCache end

struct ForwardEulerCellSolver <: AbstractCellSolver end

struct ForwardEulerCellSolverCache{T} <: AbstractCellSolverCache
    du::Vector{T}
end

function solve!(uₙ::T1, sₙ::T2, cell_model::ION, t::Float64, Δt::Float64, solver_cache::ForwardEulerCellSolverCache{T3}) where {T1, T2, T3, ION <: AbstractIonicModel}
    # Eval buffer
    @unpack du = solver_cache
    
    for i ∈ 1:length(uₙ)
        @inbounds φₘ_cell = uₙ[i]
        @inbounds s_cell  = @view sₙ[i,:]

        #TODO get x and Cₘ
        cell_rhs!(du, φₘ_cell, s_cell, nothing, t, cell_model)

        @inbounds uₙ[i] = φₘ_cell + Δt*du[1]

        # Non-allocating assignment
        @inbounds for j ∈ 1:num_states(cell_model)
            sₙ[i,j] = s_cell[j] + Δt*du[j+1]
        end
    end
end

Base.@kwdef struct AdaptiveForwardEulerReactionSubCellSolver{T} <: AbstractCellSolver
    substeps::Int = 10
    reaction_threshold::T = 0.1
end

struct AdaptiveForwardEulerReactionSubCellSolverCache{T} <: AbstractCellSolverCache
    du::Vector{T}
    substeps::Int
    reaction_threshold::T
end

function solve!(uₙ::T1, sₙ::T2, cell_model::ION, t::Float64, Δt::Float64, cache::AdaptiveForwardEulerReactionSubCellSolverCache{T3}) where {T1, T2, T3, ION <: AbstractIonicModel}
    @unpack du = cache

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
end

Base.@kwdef struct ThreadedCellSolver{SolverType<:AbstractCellSolver} <: AbstractCellSolver
    cells_per_thread::Int = 64
end

Base.@kwdef struct ThreadedCellSolverCache{CacheType<:AbstractCellSolverCache} <: AbstractCellSolverCache
    scratch::Vector{CacheType}
    cells_per_thread::Int = 64
end

function solve!(uₙ::T1, sₙ::T2, cell_model::ION, t::Float64, Δt::Float64, cache::ThreadedCellSolverCache{CacheType}) where {T1, T2, CacheType, ION <: AbstractIonicModel}
    for tid ∈ 1:cache.cells_per_thread:length(uₙ)
        tcache = cache.scratch[Threads.threadid()]
        last_cell_in_thread = min((tid+cells_per_thread),length(uₙ))
        tuₙ = @view uₙ[tid:last_cell_in_thread]
        tsₙ = @view sₙ[tid:last_cell_in_thread]
        Threads.@threads for tid ∈ 1:cells_per_thread:length(uₙ)
            solve!(tuₙ, tsₙ, cell_model, t, Δt, tcache)
        end
    end
end

######################################################

# TODO contribute back to Ferrite
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool=true) where {dim,C,T}
    cells = MeshCell[MeshCell(Ferrite.cell_to_vtkcell(typeof(cell)), Ferrite.nodes_to_vtkorder(cell)) for cell in getcells(grid)]
    coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    return vtk_grid(filename, coords, cells; compress=compress)
end

######################################################
# struct GodunovSolver <: AbstractSolver
# end

######################################################


function solve(;Δt=0.1, T=3.0, storeskip = 50, heatsolver = ImplicitEulerHeatSolver(), cellsolver = ForwardEulerCellSolver())
    heatsolvercache = ImplicitEulerHeatSolverCache(
        create_sparsity_pattern(dh),
        create_sparsity_pattern(dh),
        SparseMatrixCSR(transpose(create_sparsity_pattern(dh))),
        CgSolver(length(u₀), length(u₀), Vector{Float64}),
        zeros(ndofs(dh))
    )

    @unpack M, K, linsolver = heatsolvercache
    @timeit "assembly" assemble_global!(cellvalues, K, M, dh, epmodel)

    heatsolvercache.A = SparseMatrixCSR(transpose(M - Δt*K))

    cellsolvercache = ForwardEulerCellSolverCache(zeros(num_states(epmodel.ion)+1))

    uₙ₋₁ = u₀
    uₙ = zeros(ndofs(dh))
    sₙ = s₀

    io = ParaViewWriter("monodomain")

    @timeit "solver" for (i,t) ∈ enumerate(0.0:Δt:T)
        @info t

        @timeit "io" if (i-1) % storeskip == 0
            store_timestep!(io, t, dh, uₙ₋₁)
        end

        @timeit "heat" solve!(uₙ, uₙ₋₁, t, Δt, heatsolvercache)
        @timeit "cell" solve!(uₙ, sₙ, epmodel.ion, t, Δt, cellsolvercache)

        uₙ₋₁ .= uₙ
    end

    finalize!(io)

    @info "Done."
end
