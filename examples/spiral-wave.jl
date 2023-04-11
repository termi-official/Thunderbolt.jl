using Thunderbolt, LinearAlgebra, SparseArrays, UnPack
import Thunderbolt: AbstractIonicModel

Base.@kwdef struct FHNModel <: AbstractIonicModel
    a::Float64 = 0.1
    b::Float64 = 0.5
    c::Float64 = 1.0
    d::Float64 = 0.0
    e::Float64 = 0.01
end;

num_states(::FHNModel) = 1

function cell_rhs!(du,φₘ,s,x,t,cell_parameters::T) where {T <: AbstractIonicModel}
    dφₘ = @view du[1:1]
    ds = @view du[2:2]
    reaction_rhs!(dφₘ,φₘ,s,x,t,cell_parameters)
    state_rhs!(ds,φₘ,s,x,t,cell_parameters)
end

function reaction_rhs!(dφₘ,φₘ,s,x,t,cell_parameters::FHNModel)
    @unpack a,b,c,d,e = cell_parameters
    dφₘ .= -φₘ^3 + (1+a)*φₘ^2 -a*φₘ -s #φₘ*(1-φₘ)*(φₘ+a)
end

function state_rhs!(ds,φₘ,s,x,t,cell_parameters::FHNModel)
    @unpack a,b,c,d,e = cell_parameters
    ds .= e*(b*φₘ - c*s[1] - d)
end

epmodel = MonodomainModel(
    x->1.0,
    x->1.0,
    x->SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)),
    NoStimulationProtocol(),
    FHNModel()
)
T = 10.0
Δt = 0.1

# TODO where to put this setup?
grid = generate_grid(Quadrilateral, (80, 80), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))
# addnodeset!(grid, "ground", x-> x[2] == -0 && x[1] == -0)
dim = 2
Δt = 0.1
T = 1000
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
s₀ = zeros(ndofs(dh));
for cell in CellIterator(dh)
    _celldofs = celldofs(cell)
    ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
    for (i, coordinate) in enumerate(getcoordinates(cell))
        if coordinate[1] <= 1.25 && coordinate[2] <= 1.25
            u₀[ϕₘ_celldofs[i]] = 1.0
        end
        if coordinate[2] >= 1.25
            s₀[ϕₘ_celldofs[i]] = 0.1
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

# function assemble_global!(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, M::SparseMatrixCSC, dh::DofHandler; params::FHNParameters = FHNParameters()) where {dim}
#     n_ϕₘ = getnbasefunctions(cellvalues)
#     n_ϕₑ = getnbasefunctions(cellvalues)
#     n_s = getnbasefunctions(cellvalues)
#     ntotal = n_ϕₘ + n_ϕₑ + n_s
#     n_basefuncs = getnbasefunctions(cellvalues)
#     #We use PseudoBlockArrays to write into the right places of Ke
#     Ke = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])
#     Me = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])

#     assembler_K = start_assemble(K)
#     assembler_M = start_assemble(M)

#     #Here the block indices of the variables are defined.
#     ϕₘ▄, ϕₑ▄, s▄ = 1, 2, 3

#     #Now we iterate over all cells of the grid
#     @inbounds for cell in CellIterator(dh)
#         fill!(Ke, 0)
#         fill!(Me, 0)
#         #get the coordinates of the current cell
#         coords = getcoordinates(cell)

#         JuAFEM.reinit!(cellvalues, cell)
#         #loop over all Gauss points
#         for q_point in 1:getnquadpoints(cellvalues)
#             #get the spatial coordinates of the current gauss point
#             coords_qp = spatial_coordinate(cellvalues, q_point, coords)
#             #based on the gauss point coordinates, we get the spatial dependent
#             #material parameters
#             κₑ_loc = κₑ(coords_qp)
#             κᵢ_loc = κᵢ(coords_qp)
#             Cₘ_loc = Cₘ(coords_qp)
#             χ_loc = χ(coords_qp)
#             dΩ = getdetJdV(cellvalues, q_point)
#             for i in 1:n_basefuncs
#                 Nᵢ = shape_value(cellvalues, q_point, i)
#                 ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
#                 for j in 1:n_basefuncs
#                     Nⱼ = shape_value(cellvalues, q_point, j)
#                     ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
#                     #diffusion parts
#                     Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
#                     Ke[BlockIndex((ϕₘ▄,ϕₑ▄),(i,j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
#                     Ke[BlockIndex((ϕₑ▄,ϕₘ▄),(i,j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
#                     Ke[BlockIndex((ϕₑ▄,ϕₑ▄),(i,j))] -= (((κₑ_loc + κᵢ_loc) ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
#                     #linear reaction parts
#                     Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] -= params.a * Nᵢ * Nⱼ * dΩ
#                     Ke[BlockIndex((ϕₘ▄,s▄),(i,j))]  -= Nᵢ * Nⱼ * dΩ
#                     Ke[BlockIndex((s▄,ϕₘ▄),(i,j))]  += params.e * params.b * Nᵢ * Nⱼ * dΩ
#                     Ke[BlockIndex((s▄,s▄),(i,j))]   -= params.e * params.c * Nᵢ * Nⱼ * dΩ
#                     #mass matrices
#                     Me[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += Cₘ_loc * χ_loc * Nᵢ * Nⱼ * dΩ
#                     Me[BlockIndex((s▄,s▄),(i,j))]   += Nᵢ * Nⱼ * dΩ
#                 end
#             end
#         end

#         assemble!(assembler_K, celldofs(cell), Ke)
#         assemble!(assembler_M, celldofs(cell), Me)
#     end
#     return K, M
# end;

function solve()
    M = create_sparsity_pattern(dh);
    K = create_sparsity_pattern(dh);
    assemble_global!(cellvalues, K, M, dh, epmodel)

    #precompute decomposition
    # M(uₙ-uₙ₋₁)/Δt = Kuₙ -> (M-ΔtK)uₙ = Muₙ₋₁
    # here A := (M-ΔtK), b := Muₙ₋₁
    A = cholesky(Symmetric(M - Δt*K))

    uₙ₋₁ = u₀
    sₙ₋₁ = s₀
    uₙ = zeros(ndofs(dh))
    sₙ = zeros(ndofs(dh))
    du_cell = zeros(num_states(epmodel.ion)+1)

    pvd = paraview_collection("monodomain.pvd");

    for (i,t) ∈ enumerate(0.0:Δt:T)
        @info t
        
        if (i-1) % 10 == 0
            vtk_grid("monodomain-$t.vtu", dh) do vtk
                vtk_point_data(vtk,dh,uₙ₋₁)
                vtk_save(vtk)
                pvd[t] = vtk
            end
        end

        # Heat Solver - Implicit Euler
        b = M*uₙ₋₁
        uₙ = A\b

        # Cell Solver - Explicit Euler
        for i ∈ 1:ndofs(dh)
            φₘ_cell = uₙ[i]
            s_cell  = sₙ₋₁[i]

            #TODO get x and Cₘ
            cell_rhs!(du_cell, φₘ_cell, s_cell, [], t, epmodel.ion)

            uₙ[i] = φₘ_cell + Δt*du_cell[1]
            sₙ[i] = s_cell + Δt*du_cell[2]
        end

        uₙ₋₁ .= uₙ
        sₙ₋₁ .= sₙ
    end
    vtk_save(pvd);
    @info "Done."
end
