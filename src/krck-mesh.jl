# Mesh based on "Efficient multi-level hp-finite elements in arbitrary dimensions." (2022)
using StaticArrays

abstract type AbstractKRCKElement{sdim,N,M} <: AbstractCell{sdim,N,M} end

struct KRCKElement{sdim,N,M} <: AbstractKRCKElement{sdim,N,M}
    active_nodes::MVector{N, Bool}
    # parent::Int 
    face_neighbors::MVector{M, Int}
    level::Int
    # children::Union{Nothing, KRCKElement{sdim,N,M}}
end

struct KRCKMesh{sdim, E <: KRCKElement, T <: Real} <: Ferrite.AbstractGrid{sdim}
    root_node_coords::Vector{Ferrite.Node{sdim, T}}
    root_elements::Vector{E}
end

const KRCKQuad2D = KRCKElement{2,4,4}

function refine!(mesh::KRCKMesh, element_idx::Int)
    element_to_refine = mesh.elements[element_idx]
    @assert element_to_refine.is_leaf == true
    _refine!(mesh, element_to_refine)
end

#
# D---+---C
# | 4 | 3 |
# +---+---+
# | 1 | 2 |
# A---+---B
#
function _refine!(mesh::KRCKMesh, element_idx::Int, element_to_refine::KRCKQuad2D)
    # ... ?
end