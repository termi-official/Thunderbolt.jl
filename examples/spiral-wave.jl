using Thunderbolt, LinearAlgebra, SparseArrays, UnPack
import Thunderbolt: AbstractIonicModel

using TimerOutputs, BenchmarkTools

using Krylov

using SparseMatricesCSR, ThreadedSparseCSR
ThreadedSparseCSR.multithread_matmul(PolyesterThreads())

Base.@kwdef struct ParametrizedFHNModel{T} <: AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T = T(0.01)
end;

const FHNModel = ParametrizedFHNModel{Float64};

num_states(::ParametrizedFHNModel{T}) where{T} = 1

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

epmodel = MonodomainModel(
    x->1.0,
    x->1.0,
    x->SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)),
    NoStimulationProtocol(),
    FHNModel()
)

# TODO where to put this setup?
grid = generate_grid(Quadrilateral, (512, 512), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))
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

# Solver
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

function step_cells!(uₙ::T1, sₙ::T2, sₙ₋₁::T2, cell_model::ION, t::Float64, Δt::Float64, du_cell_arr::T3) where {T1, T2, T3, ION <: AbstractIonicModel}
    cells_per_thread = 64
    # @timeit "cell" for tid ∈ 1:cells_per_thread:length(uₙ)
    # # @timeit "cell" Threads.@threads for tid ∈ 1:cells_per_thread:length(uₙ)
    #     du_cell = du_cell_arr[Threads.threadid()]
    #     upper_bound = min((tid+cells_per_thread),length(uₙ))
    #     for i ∈ tid:upper_bound
    @timeit "cell" for i ∈ 1:length(uₙ)
        du_cell = du_cell_arr[Threads.threadid()]
            @inbounds φₘ_cell = uₙ[i]
            @inbounds s_cell  = @view sₙ₋₁[i,:]

            #TODO get x and Cₘ
            cell_rhs!(du_cell, φₘ_cell, s_cell, nothing, t, cell_model)

            @inbounds uₙ[i] = φₘ_cell + Δt*du_cell[1]
            # Non-allocating assignment
            @inbounds for j ∈ 1:num_states(cell_model)
                sₙ[i,j] = s_cell[j] + Δt*du_cell[j+1]
            end
    #    end
    end
end

# TODO contribute back to Ferrite
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool=true) where {dim,C,T}
    cells = MeshCell[MeshCell(Ferrite.cell_to_vtkcell(typeof(cell)), Ferrite.nodes_to_vtkorder(cell)) for cell in getcells(grid)]
    coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    return vtk_grid(filename, coords, cells; compress=compress)
end

function solve(;Δt=0.1, T=3.0, vtkskip = 50)
    cells_per_thread = 64

    M = create_sparsity_pattern(dh);
    K = create_sparsity_pattern(dh);
    @timeit "assembly" assemble_global!(cellvalues, K, M, dh, epmodel)

    A = SparseMatrixCSR(transpose(M - Δt*K))
    # Pl = JacobiPreconditioner(A) # Jacobi preconditioner

    uₙ₋₁ = u₀
    sₙ₋₁ = s₀
    uₙ = zeros(ndofs(dh))
    b  = zeros(ndofs(dh))
    sₙ = zeros(ndofs(dh),num_states(epmodel.ion))
    du_cell_arr = [zeros(num_states(epmodel.ion)+1) for t ∈ 1:Threads.nthreads()]

    pvd = paraview_collection("monodomain.pvd");
    
    linsolver = CgSolver(length(u₀), length(u₀), Vector{Float64})

    @timeit "solver" for (i,t) ∈ enumerate(0.0:Δt:T)
        @info t

        @timeit "io" if (i-1) % vtkskip == 0
            vtk_grid("monodomain-$t.vtu", dh) do vtk
                vtk_point_data(vtk,dh,uₙ₋₁)
                vtk_save(vtk)
                pvd[t] = vtk
            end
        end

        # Heat Solver - Implicit Euler
        @timeit "rhs" mul!(b, M, uₙ₋₁)
        @timeit "Axb" begin
            # Krylov.cg!(linsolver, A, b, uₙ₋₁, M=Pl, ldiv=true)
            Krylov.cg!(linsolver, A, b, uₙ₋₁)
            uₙ = linsolver.x
        end

        # Cell Solver - Explicit Euler
        step_cells!(uₙ, sₙ, sₙ₋₁, epmodel.ion, t, Δt, du_cell_arr)
        #@timeit "cell" Threads.@threads for tid ∈ 1:cells_per_thread:ndofs(dh)
        # @timeit "cell" for i ∈ 1:ndofs(dh)
        #     du_cell = du_cell_arr[Threads.threadid()]
        #     # for i ∈ tid:min((tid+cells_per_thread-1),ndofs(dh))
        #         φₘ_cell = uₙ[i]
        #         s_cell  = @view sₙ₋₁[i,1:num_states(epmodel.ion)]

        #         #TODO get x and Cₘ
        #         cell_rhs!(du_cell, 0.0, 0.0, nothing, t, epmodel.ion)

        #         uₙ[i] = φₘ_cell + Δt*du_cell[1]
        #         for j ∈ 1:num_states(epmodel.ion)
        #             sₙ[i,j] = s_cell[j] + Δt*du_cell[j+1]
        #         end
        #     # end
        # end

        uₙ₋₁ .= uₙ
        sₙ₋₁ .= sₙ
    end
    vtk_save(pvd);
    @info "Done."
end
