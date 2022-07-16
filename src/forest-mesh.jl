using StaticArrays

# Unique index of a cell in a forest mesh.
struct ForestElementIndex
    root_idx::Int
    path::Vector{Int}
end

import Base.==
function ==(A::ForestElementIndex, B::ForestElementIndex)
    return A.root_idx == B.root_idx && A.path == B.path
end

# Elements whose refinement can be tracked to allow derefinement.
# `sdim` is the spatial dimension of the ambient space
# `N` is the number of nodes in the element
# `M` is the number of (sdim-1)-dimensional entities in the element
abstract type RootElement{sdim,N,M} <: Ferrite.AbstractCell{sdim,N,M} end
abstract type TreeElement{sdim,N,M} <: Ferrite.AbstractCell{sdim,N,M} end

# abstract type TensorProductTreeElement{dim::Int, order::Int} <: TreeElement{dim,(order+1)^dim,2*dim} end
# isrefined(::TensorProductTreeElement{dim,order}) = true where {dim,order}

# # Not yet refined tensor product element
# struct TensorProductLeafElement{dim, order} <: TensorProductTreeElement{dim, order}
# end
# isrefined(::TensorProductLeafElement{dim,order}) = false where {dim,order}

# # Isotropically refined tensor product element into 2^dim elements
# struct IsotropicRefinedTensorProductElement{dim, order} <: TensorProductTreeElement{dim, order}
#     children::NTuple{2^dim, TensorProductTreeElement{dim, order}}
# end
# # function refine_isotropic(element::IsotropicRefinedTensorProductElement, which::Int)
# #     element.children[] IsotropicRefinedTensorProductElement{dim, order}(ntuple(x->TensorProductLeafElement(),2^dim))
# # end

# # Elements which are tensor products of lines.
# struct TensorProductRootElement{dim,order} <: RootElement{dim,dim+1,dim+1} where {dim,order}
#     nodes::NTuple{(order+1)^dim,Int}
#     element::TensorProductTreeElement
# end

# function TensorProductRootElement{dim,order}(nodes::NTuple{(order+1)^dim,Int}) where TensorProductRootElement{dim,order}
#     return TensorProductRootElement{dim,order}(nodes,TensorProductLeafElement{dim, order})
# end

# Elements which are dim-dimensional simplices.
# struct SimplicalRootElement{dim} <: RootElement{dim,2^dim,2*dim}
#     nodes::NTuple{2^dim,Int}
# end

# We can't do above.... https://github.com/JuliaLang/julia/issues/18466
abstract type TensorProductTreeElement{sdim,N,M} <: TreeElement{sdim,N,M} end

isleaf(element::TreeElement{sdim,N,M}) where {sdim,N,M} = !isrefined(element)

mutable struct TensorProductRootElement{sdim,N,M} <: RootElement{sdim,N,M}
    nodes::NTuple{N,Int}
    element::TensorProductTreeElement{sdim,N,M}
end

isrefined(::TensorProductTreeElement{sdim,N,M}) where {sdim,N,M} = true

struct RefinedLineElement <: TensorProductTreeElement{1, 2, 2}
    children::SizedVector{2, TensorProductTreeElement{1, 2, 2}}
end

function RefinedLineElement()
    return RefinedLineElement(ntuple(x->Thunderbolt.LeafLineElement(),2))
end
function refine_isotropic!(element::RefinedLineElement, which_child::Int)
    element.children[which_child] = RefinedLineElement()
end

struct LeafLineElement <: TensorProductTreeElement{1, 2, 2}
end

isrefined(::LeafLineElement) = false

refine_isotropic(::LeafLineElement) = RefinedLineElement()

function derefine!(element::RefinedLineElement, which_child::Int)
    element.children[which_child] = LeafLineElement()
end

###########################
# Utils
###########################
function to_root_element(cell::Line)
    return TensorProductRootElement{1,2,2}(cell.nodes, LeafLineElement())
end

###########################


# Topology information for possibly non-conforming forest meshes.
struct ForestTopology <: Ferrite.AbstractTopology
    root_node_to_cell::Vector{Vector{Int}}
    root_cell_neighbors::Vector{Ferrite.EntityNeighborhood{CellIndex}}
end

# Mesh whose elements can be refined and derefined. The resulting mesh may be non-conforming.
# - `sdim`: spatial dimensional
# - `C`: element type
# - `T`: type for the node
struct ForestMesh{sdim,CR<:RootElement,T<:Real} <: Ferrite.AbstractGrid{sdim}
    # Cells of the root elements
    cells::Vector{CR}

    # Nodes of the root elements
    nodes::Vector{Ferrite.Node{sdim,T}}

    # Sets defined on the root elements
    cellsets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{Ferrite.FaceIndex}}
    edgesets::Dict{String,Set{Ferrite.EdgeIndex}}
    vertexsets::Dict{String,Set{Ferrite.VertexIndex}}
end

function ForestMesh(grid::Grid{sdim,C,T}) where {sdim,C,T}

    return ForestMesh(to_root_element.(grid.cells),grid.nodes,grid.cellsets,grid.facesets,grid.edgesets,grid.vertexsets)
end

function get_element_parent(mesh::ForestMesh{sdim, C, T}, tree_idx::ForestElementIndex) where {sdim, C, T}
    element = mesh.cells[tree_idx.root_idx].element
    for childidx ∈ tree_idx.path[1:(end-1)]
        element = element.children[childidx]
    end
    return element
end


function get_element(mesh::ForestMesh{sdim, C, T}, tree_idx::ForestElementIndex) where {sdim, C, T}
    element = mesh.cells[tree_idx.root_idx]
    for childidx ∈ tree_idx.path
        element = element.children[childidx]
    end
    return element
end

function refine_isotropic!(mesh::ForestMesh{sdim, C, T}, tree_idx::ForestElementIndex) where {sdim, C, T}
    if isempty(tree_idx.path)
        mesh.cells[tree_idx.root_idx].element = refine_isotropic(mesh.cells[tree_idx.root_idx].element)
    else
        element = get_element_parent(mesh, tree_idx)
        refine_isotropic!(element, tree_idx.path[end])
    end
end

function derefine!(mesh::ForestMesh, tree_idx::ForestElementIndex)
    element = get_element_parent(mesh, tree_idx)
    derefine!(element, tree_idx.path[end])
end

############################

# `sdim`: spatial dimension
# `CR`: type for the root elements in the grid
# `CT`: type for the tree elements in the grid
# `T`: type for the node coordinate components of the grid
mutable struct ForestIterator{sdim,CR<:RootElement,T}
    grid::ForestMesh{sdim,CR,T}
    element_stack::Vector{TreeElement}
    current_element_idx::ForestElementIndex

    function ForestIterator(grid::ForestMesh{sdim,CR,T}) where {sdim,CR,T}
        return new{sdim,CR,T}(grid, [], ForestElementIndex(0, []))
    end
end

function Base.iterate(fi::ForestIterator)
    if isempty(fi.grid.cells)
        return nothing
    end

    fi.current_element_idx = ForestElementIndex(1, [])
    fi.element_stack = [fi.grid.cells[1].element]
    return (fi, 1)
end

function Base.iterate(fi::ForestIterator, state)
    # Get next element index
    if isrefined(fi.element_stack[end])
        push!(fi.element_stack, fi.element_stack[end].children[1])
        push!(fi.current_element_idx.path, 1)
    else # next element is leaf
        while true
            if length(fi.element_stack) == 1
                fi.current_element_idx = ForestElementIndex(fi.current_element_idx.root_idx+1, [])
                if fi.current_element_idx.root_idx > length(fi.grid.cells)
                    return nothing
                end
                fi.element_stack = [fi.grid.cells[fi.current_element_idx.root_idx].element]
                break
            else
                next_child_idx = fi.current_element_idx.path[end]+1
                fi.element_stack = fi.element_stack[1:(end-1)]
                if next_child_idx <= length(fi.element_stack[end].children) # next child exists on current level
                    fi.current_element_idx.path[end] = next_child_idx
                    push!(fi.element_stack , fi.element_stack[end].children[next_child_idx])
                    break
                else
                    fi.current_element_idx = ForestElementIndex(fi.current_element_idx.root_idx, fi.current_element_idx.path[1:(end-1)])
                end
            end
        end
    end

    return (fi, state+1)
end

############################

function WriteVTK.vtk_grid(filename::AbstractString, mesh::ForestMesh{sdim,C,T}; compress::Bool=true) where {sdim,C,T}
    cells = MeshCell[]
    coords = T[]
    for it in ForestIterator(mesh)
        if isleaf(it.element_stack[end])

        end
    end
    return vtk_grid(filename, coords, cells; compress=compress)
end
