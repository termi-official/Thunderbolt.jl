using Thunderbolt, LinearAlgebra, SparseArrays, UnPack
import Thunderbolt: AbstractIonicModel

using TimerOutputs, BenchmarkTools

using Krylov

import LinearAlgebra: mul!

using SparseArrays: AbstractSparseMatrixCSC #,AbstractSparseMatrixCSR?
using SparseMatricesCSR #, ThreadedSparseCSR
# ThreadedSparseCSR.multithread_matmul(PolyesterThreads())

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
struct ImplicitEulerHeatSolver
end

mutable struct ImplicitEulerHeatSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
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

@doc raw"""
    AssembledMassOperator{MT, CV}

Assembles the matrix associated to the bilinearform ``a(u,v) = -\int v(x) u(x) dx`` for ``u,v`` from the same function space.
"""
struct AssembledMassOperator{MT, CV}
    M::MT
    cv::CV
end
mul!(b, M::AssembledMassOperator{MT, CV}, uₙ₋₁) where {MT, CV} = mul!(b, M.M, uₙ₋₁)

function setup_mass_operator!(bifi::AssembledMassOperator{MT, CV}, dh::DH) where {MT, CV, DH}
    @unpack M, cv = bifi

    assembler_M = start_assemble(M)

    n_basefuncs = getnbasefunctions(cv)
    Mₑ = zeros(n_basefuncs, n_basefuncs)

    # TODO kernel
    @inbounds for cell in CellIterator(dh)
        fill!(Mₑ, 0)

        reinit!(cv, cell)

        for q_point in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q_point)
            for i in 1:n_basefuncs
                Nᵢ = shape_value(cv, q_point, i)
                for j in 1:n_basefuncs
                    Nⱼ = shape_value(cv, q_point, j)
                    Mₑ[i,j] += Nᵢ * Nⱼ * dΩ
                end
            end
        end

        assemble!(assembler_M, celldofs(cell), Mₑ)
    end
end

using StaticArrays
struct PlanarDiffusionTensorCoefficient{MSC}
    microstructure_cache::MSC
    conductivities::SVector{2}
end

function Thunderbolt.evaluate_coefficient(coeff::PlanarDiffusionTensorCoefficient{MSC}, cell_id::Int, ξ::Vec{rdim}, t::Float64=0.0) where {MSC, rdim}
    f₀, s₀ = directions(coeff.microstructure_cache, cell_id, ξ, t)
    return coeff.conductivities[1] * f₀ ⊗ f₀ + coeff.conductivities[2] * s₀ ⊗ s₀
end

struct ConductivityToDiffusivityCoefficient{DTC, CC, STVC}
    conductivity_tensor_coefficient::DTC
    capacitance_coefficient::CC
    χ_coefficient::STVC
end

function Thunderbolt.evaluate_coefficient(coeff::ConductivityToDiffusivityCoefficient{DTC, CC, STVC}, cell_id::Int, ξ::Vec{rdim}, t::Float64=0.0) where {DTC, CC, STVC, rdim}
    κ = evaluate_coefficient(coeff.conductivity_tensor_coefficient, cell_id, ξ)
    Cₘ = evaluate_coefficient(coeff.capacitance_coefficient, cell_id, ξ)
    χ = evaluate_coefficient(coeff.χ_coefficient, cell_id, ξ)
    return κ/(Cₘ*χ)
end

@doc raw"""
    AssembledDiffusionOperator{MT, DT, CV}

Assembles the matrix associated to the bilinearform ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct AssembledDiffusionOperator{MT, DT, CV}
    K::MT
    diffusion_coefficient::DT
    cv::CV
end
mul!(b, K::AssembledDiffusionOperator{MT, DT, CV}, uₙ₋₁) where {MT, DT, CV} = mul!(b, K.K, uₙ₋₁)

function setup_diffusion_operator!(bifi::AssembledDiffusionOperator{MT, CV}, dh::DH) where {MT, CV, DH}
    @unpack K, diffusion_coefficient, cv = bifi

    n_basefuncs = getnbasefunctions(cv)
    Kₑ = zeros(n_basefuncs, n_basefuncs)

    assembler_K = start_assemble(K)

    # TODO kernel
    @inbounds for cell in CellIterator(dh)
        fill!(Kₑ, 0)

        reinit!(cv, cell)

        for q_point in 1:getnquadpoints(cv)
            ξ = cv.qr.points[q_point]
            D_loc = evaluate_coefficient(diffusion_coefficient, cellid(cell), ξ)
            #based on the gauss point coordinates, we get the spatial dependent
            #material parameters
            dΩ = getdetJdV(cv, q_point)
            for i in 1:n_basefuncs
                ∇Nᵢ = shape_gradient(cv, q_point, i)
                for j in 1:n_basefuncs
                    ∇Nⱼ = shape_gradient(cv, q_point, j)
                    Kₑ[i,j] -= ((D_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                end
            end
        end

        assemble!(assembler_K, celldofs(cell), Kₑ)
    end
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::ImplicitEulerHeatSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    cache.A = SystemMatrixType(cache.M.M - Δt*cache.K.K) # TODO FIXME make me generic
    cache.Δt_last = Δt
end

# Optimized version for CSR matrices
#TODO where is AbstractSparseMatrixCSR ?
function implicit_euler_heat_solver_update_system_matrix!(cache::ImplicitEulerHeatSolverCache{SolutionType, SMT1, SMT2, SMT2, LinSolverType, RHSType}, Δt) where {SolutionType, SMT1 <: SparseMatrixCSR, SMT2 <: AbstractSparseMatrixCSC, LinSolverType, RHSType}
    cache.A = SparseMatrixCSR(transpose(cache.M.M - Δt*cache.K.K)) # TODO FIXME make me generic
end

# Performs a backward Euler step
function perform_step!(problem, cache::ImplicitEulerHeatSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, t, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
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
end

# TODO decouple the concept ImplicitEuler from TransientHeatProblem again
function setup_solver_caches(problem #= ::TransientHeatProblem =#, solver::ImplicitEulerHeatSolver)
    @unpack dh = problem
    @assert length(dh.field_names) == 1 # TODO relax this assumption, maybe.
    ip = dh.subdofhandlers[1].field_interpolations[1]
    order = Ferrite.getorder(ip)
    cv = CellValues(
        QuadratureRule{RefQuadrilateral}(2*order), # TODO how to pass this one down here?
        ip
    )
    cache = ImplicitEulerHeatSolverCache(
        zeros(ndofs(dh)),
        zeros(ndofs(dh)),
        # TODO How to choose the exact operator types here?
        #      Maybe via some parameter in ImplicitEulerHeatSolver?
        AssembledMassOperator(
            create_sparsity_pattern(dh),
            cv
        ),
        AssembledDiffusionOperator(
            create_sparsity_pattern(dh),
            problem.diffusion_tensor_field,
            cv
        ),
        create_sparsity_pattern(dh),
        # TODO this via LinearSolvers.jl
        CgSolver(ndofs(dh), ndofs(dh), Vector{Float64}),
        zeros(ndofs(dh)),
        0.0
    )

    # TODO where does this belong?
    setup_mass_operator!(cache.M, dh)
    setup_diffusion_operator!(cache.K, dh)

    return cache
end

######################################################

abstract type AbstractCellSolver end
abstract type AbstractCellSolverCache end

struct ForwardEulerCellSolver <: AbstractCellSolver
end

mutable struct ForwardEulerCellSolverCache{VT, MT} <: AbstractCellSolverCache
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
end

# TODO decouple the concept ForwardEuler from "CellProblem"
function setup_solver_caches(problem, solver::ForwardEulerCellSolver)
    @unpack npoints = problem # TODO what is a good abstraction layer over this?
    return ForwardEulerCellSolverCache(
        zeros(1+num_states(problem.ode)),
        zeros(npoints),
        zeros(npoints, num_states(problem.ode))
    )
end

Base.@kwdef struct AdaptiveForwardEulerReactionSubCellSolver{T} <: AbstractCellSolver
    substeps::Int = 10
    reaction_threshold::T = 0.1
end

struct AdaptiveForwardEulerReactionSubCellSolverCache{VT, T} <: AbstractCellSolverCache
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
end

# Base.@kwdef struct ThreadedCellSolver{SolverType<:AbstractCellSolver} <: AbstractCellSolver
#     cells_per_thread::Int = 64
# end

# Base.@kwdef struct ThreadedCellSolverCache{CacheType<:AbstractCellSolverCache} <: AbstractCellSolverCache
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

######################################################

# TODO contribute back to Ferrite
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool=true) where {dim,C,T}
    cells = MeshCell[MeshCell(Ferrite.cell_to_vtkcell(typeof(cell)), Ferrite.nodes_to_vtkorder(cell)) for cell in getcells(grid)]
    coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    return vtk_grid(filename, coords, cells; compress=compress)
end

######################################################
struct ReactionDiffusionSplit{MODEL}
    model::MODEL
end

struct TransientHeatProblem{DTF, ST, DH}
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
end

struct PointwiseODEProblem{ODEDT}
    npoints::Int
    ode_decider::ODEDT
end

struct PointwiseHomogeneousODEProblem{ODET}
    npoints::Int
    ode::ODET
end

perform_step!(problem::PointwiseHomogeneousODEProblem{ODET}, cache::CT, t::Float64, Δt::Float64) where {ODET, CT} = perform_step!(problem.ode, t, Δt, cache)

struct SplitProblem{APT, BPT}
    A::APT
    B::BPT
end

struct LTGOSSolver{AST,BST}
    A_solver::AST
    B_solver::BST
end

struct LTGOSSolverCache{ASCT, BSCT}
    A_solver_cache::ASCT
    B_solver_cache::BSCT
end

# TODO add guidance with helpers like
#   const QuGarfinkel1999Solver = SMOSSolver{AdaptiveForwardEulerReactionSubCellSolver, ImplicitEulerHeatSolver}

"""
    transfer_fields!(A, A_cache, B, B_cache)

The entry point to prepare the field evaluation for the time step of problem B, given the solution of problem A.
The default behavior assumes that nothing has to be done, because both problems use the same unknown vectors for the shared parts.
"""
transfer_fields!(A, A_cache, B, B_cache)

transfer_fields!(A, A_cache::ImplicitEulerHeatSolverCache, B, B_cache::ForwardEulerCellSolverCache) = nothing
transfer_fields!(A, A_cache::ForwardEulerCellSolverCache, B, B_cache::ImplicitEulerHeatSolverCache) = nothing

function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{AST,BST}) where {APT,BPT,AST,BST}
    return LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver),
        setup_solver_caches(problem.B, solver.B_solver),
    )
end

function setup_solver_caches(problem::SplitProblem{APT, BPT}, solver::LTGOSSolver{ImplicitEulerHeatSolver,BST}) where {APT,BPT,BST}
    cache = LTGOSSolverCache(
        setup_solver_caches(problem.A, solver.A_solver),
        setup_solver_caches(problem.B, solver.B_solver),
    )
    cache.B_solver_cache.uₙ = cache.A_solver_cache.uₙ
    return cache
end

# Lie-Trotter-Godunov step to advance the problem split into A and B from a given initial condition
function perform_step!(problem::SplitProblem{APT, BPT}, cache::LTGOSSolverCache{ASCT, BSCT}, t, Δt) where {APT, BPT, ASCT, BSCT}
    # We start by setting the initial condition for the step of problem A from the solution in B.
    transfer_fields!(problem.B, cache.B_solver_cache, problem.A, cache.A_solver_cache)
    # Then the step for A is executed
    perform_step!(problem.A, cache.A_solver_cache, t, Δt)
    # This sets the initial condition for problem B
    transfer_fields!(problem.A, cache.A_solver_cache, problem.B, cache.B_solver_cache)
    # Then the step for B is executed
    perform_step!(problem.B, cache.B_solver_cache, t, Δt)
    # This concludes the time step
    return nothing
end

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
        @info t
        @timeit "solver" perform_step!(problem, solver_cache, t, Δt)

        callback(t, problem, solver_cache)

        # TODO Δt adaption
        t += Δt
    end

    @info T
    @timeit "solver" perform_step!(problem, solver_cache, T, T-t)
    callback(t, problem, solver_cache)

    return true
end
#####################################################

mutable struct IOCallback{IO}
    io::IO
end

function (iocb::IOCallback{ParaViewWriter{PVD}})(t, problem::SplitProblem, solver_cache) where {PVD}
    store_timestep!(iocb.io, t, problem.A.dh.grid)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.A_solver_cache.uₙ, :φₘ)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.B_solver_cache.sₙ[1:length(c.A_solver_cache.uₙ)], :s)
    Thunderbolt.finalize_timestep!(iocb.io, t)
end

#####################################################
# TODO what exactly is the job here? How do we know where to write and what to iterate?
function setup_initial_condition!(problem::SplitProblem, cache, initial_condition)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    u₀, s₀ = initial_condition(problem)
    cache.B_solver_cache.uₙ .= u₀ # Note that the vectors in the caches are connected
    cache.B_solver_cache.sₙ .= s₀
    return nothing
end

function Thunderbolt.semidiscretize(split::ReactionDiffusionSplit{MonodomainModel{A,B,C,D,E}}, discretization::FiniteElementDiscretization, grid::Thunderbolt.AbstractGrid) where {A,B,C,D,E}
    epmodel = split.model

    ets = elementtypes(grid)
    @assert length(ets) == 1

    ip = getinterpolation(discretization.interpolations[:φₘ], getcells(grid, 1))
    dh = DofHandler(grid)
    push!(dh, :ϕₘ, ip)
    close!(dh);

    #
    semidiscrete_problem = SplitProblem(
        TransientHeatProblem(
            ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
            epmodel.stim,
            dh
        ),
        PointwiseHomogeneousODEProblem(
            # TODO epmodel.Cₘ(x) and coordinates
            ndofs(dh),
            epmodel.ion
        )
    )

    return semidiscrete_problem
end

######################################################
function spiral_wave_initializer(problem::SplitProblem)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    dh = problem.A.dh
    ionic_model = problem.B.ode
    u₀ = zeros(ndofs(dh));
    s₀ = zeros(ndofs(dh), num_states(ionic_model));
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        # TODO get coordinate via coordinate_system
        coordinates = getcoordinates(cell)
        for (i, (x₁, x₂)) in enumerate(coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                u₀[ϕₘ_celldofs[i]] = 1.0
            end
            if x₂ >= 1.25
                s₀[ϕₘ_celldofs[i],1] = 0.1
            end
        end
    end
    return u₀, s₀
end

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5))),
    NoStimulationProtocol(),
    FHNModel()
)

mesh = generate_mesh(Quadrilateral, (2^6, 2^6), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)

solver = LTGOSSolver(
    ImplicitEulerHeatSolver(),
    ForwardEulerCellSolver()
)

# Idea: We want to solve a semidiscrete problem, with a given compatible solver, on a time interval, with a given initial condition
# TODO iterator syntax
solve(
    problem,
    solver,
    1.0,
    (0.0, 1000.0),
    spiral_wave_initializer,
    IOCallback(ParaViewWriter("test"))
)
