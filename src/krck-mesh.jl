# Mesh based on "Efficient multi-level hp-finite elements in arbitrary dimensions." (2022)
using StaticArrays

abstract type AbstractTreeElement{ref_shape} <: AbstractCell{ref_shape} end

@enumx RBSTreeLineElementPlacement Iso1 Iso2

struct RBSTreeLineElement <: AbstractTreeElement{RefLine}}
    parent::Int
    level::Int
    relative_position::RBSTreeLineElementPlacement
end

@enumx RBSTreeQuadrilateralElementPlacement Iso11 Iso12 Iso21 Iso22 #AnisoX1 AnisoX2 AnisoY1 AnisoY2

struct RBSTreeQuadrilateralElement <: AbstractCell{RefQuadrilateral}
    active_vertices::MVector{4, Bool}
    active_faces::MVector{4, Bool}
    level::NTuple{2, Int}
    face_neighbors::MVector{4, Bool}
end

@enumx RBSTreeQuadrilateralElementPlacement Iso111 Iso112 Iso121 Iso122 Iso211 Iso212 Iso221 Iso222 #AnisoX1 AnisoX2 AnisoY1 AnisoY2

struct RBSTreeHexahedronElement <: AbstractCell{RefHexahedron}
    active_vertices::MVector{8, Bool}
    active_faces::MVector{6, Bool}
    active_edges::MVector{12, Bool}
    level::NTuple{3, Int}
    face_neighbors::MVector{6, Bool}
end

struct KRCKMesh{sdim, E <: KRCKElement, T <: Real} <: Ferrite.AbstractGrid{sdim}
    root_node_coords::Vector{Ferrite.Node{sdim, T}}
    root_elements::Vector{E}
end

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
