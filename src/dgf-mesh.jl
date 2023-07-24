# Optimized DG Forest Mesh Data Type
using StaticArrays

"""
    DGForestGrid{sdim,CT,T}

Forest of trees data structure optimized for DG applications.
"""
struct DGForestGrid{sdim,CT,T}
    root_nodes::Vector{Vec{sdim,T}}
    root_cells::Vector{CT}
end

struct DGForestCellIterator
    leaves_only::Bool
    current_root_cell_index::Int
end

function CellIterator(grid::DGForestGrid{sdim,CT,T}, leaves_only = true)
    return nothing
end

function refine_cells_isotropic!(grid::DGForestGrid{sdim,CT,T}, cells_to_refine_by_index::Vector{Int})
    if empty(cells_to_refine_by_index) return nothing
    @debug @assert issorted(cells_to_refine_by_index)
    return nothing
end

"""
    find_tree(grid::DGForestGrid{sdim,CT,T}, cell_index::Int)

Binary search to find a tree in the forst containing the cell.
"""
function find_tree(grid::DGForestGrid{sdim,CT,T}, cell_index::Int)
    @debug @assert cell_index <= last_cell_index(grid.root_cells)

    lower_cell_index = 1
    upper_cell_index = length(root_cells)
    next_root_cell_index = Int(ceil((upper_cell_index+lower_cell_index-1)/2))
    next_root_cell = grid.root_cells[next_root_cell_index]
    while cell_index ∉ cell_range(next_root_cell)
        if cell_index < first_cell_index(next_root_cell)
            upper_cell_index = next_root_cell_index
        else
            lower_cell_index = next_root_cell_index
        end
        next_root_cell_index = Int(ceil((upper_cell_index+lower_cell_index-1)/2))
        next_root_cell = grid.root_cells[next_root_cell_index]
    end
    return next_root_cell
end


"""
"""
mutable struct DGFQuadrilateralTreeCell <: AbstractCell{RefQuadrilateral} 
    const root_cell::DGFQuadrilateralCell
    const nodes::SVector{4, Int}
    num_cells_in_tree::Int
end

function DGFQuadrilateralTreeCell(cell_index::Int, quad::Quadrilateral, root_top::ExclusiveTopology)
    DGFQuadrilateralTreeCell(
        DGFQuadrilateralCell(
            0,
            cell_index,
            0,
            ntuple(i-> getneighborhood(root_top, FaceIndex(cell_index, i))[1], 4),
            0,
            nothing
            )
        ),
        quad.nodes,
        1
    )
end

first_cell_index(tree_cell) = tree_cell.root_cell.self_index
last_cell_index(tree_cell) = tree_cell.root_cell.self_index+tree_cell.num_cells_in_tree-1
cell_range(tree_cell) = first_cell_index(tree_cell):last_cell_index(tree_cell)

"""
    find_cell(tree::DGFQuadrilateralTreeCell, cell_index::Int)

Search to find a cell in the tree.
"""
function find_cell(tree::DGFQuadrilateralTreeCell, cell_index::Int)
    if tree.root_cell.self_index == cell_index
        return tree.root_cell
    else
        return find_cell(tree.root_cell, cell_index)
    end
end

function find_cell(cell::DGFQuadrilateralCell, cell_index::Int)
    @debug @assert cell.children !== nothing
    if cell_index == cell.children[1].self_index
        return cell.children[1]
    elseif cell_index < cell.children[2].self_index
        return find_cell(cell.children[1], cell_index)
    elseif cell_index == cell.children[2].self_index
        return cell.children[2]
    elseif cell_index < cell.children[3].self_index
        return find_cell(cell.children[2], cell_index)
    elseif cell_index == cell.children[3].self_index
        return cell.children[3]
    elseif cell_index < cell.children[4].self_index
        return find_cell(cell.children[3], cell_index)
    elseif cell_index == cell.children[4].self_index
        return cell.children[4]
    else
        return find_cell(cell.children[4], cell_index)
    end
end

"""
    DGFQuadrilateralCell

Cell described as a node in a quad-tree, optimized for DG applications.

* `parent_index::Int` forest cell index of the parent cell. If this index is 0, then this cell is a root cell
* `self_index::Int` forest tree cell index of the cell itself
* `self_local_index::Int` local index of this cell in the parent cell
* `face_neighbor_indices::Int` "coarsest leaf" indices
* `level::Int` current depth of the cell
* `children::Union{Nothing, SVector{DGFQuadrilateralCell, 4}}` child cells if the cell is refined, nothing if the cell is a leaf
"""
mutable struct DGFQuadrilateralCell
    parent_index::Int
    self_index::Int
    const self_local_index::Int
    face_neighbor_indices::MVector{4, FaceIndex}
    const level::Int
    children::Union{Nothing, SVector{DGFQuadrilateralCell, 4}}
end

is_leaf(cell) = cell.children === nothing

num_children_isotropic(::DGFQuadrilateralCell) = 4

# TODO: Correction of the face neighbors
#   D-------C
#   |   3   |
#   |4     2|
#   |   1   |
#   A-------B
#      to
#   D---+---C
#   | 3 | 4 |
#   +---+---+
#   | 1 | 2 |
#   A---+---B
#
function refine_cell_isotropic!(cell::DGFQuadrilateralCell, first_new_cell_index::Int)
    # Check for multiple refinements
    @debug @assert is_leaf(cell.children)

    # Compute child indices
    c1idx = first_new_cell_index+0
    c2idx = first_new_cell_index+1
    c3idx = first_new_cell_index+2
    c4idx = first_new_cell_index+3

    # Create children
    cell.children = (
        # Child 1
        DGFQuadrilateralCell(
            cell.self_index,
            c1idx,
            1,
            (
                cell.face_neighbors[1],
                FaceIndex(e2idx,4),
                FaceIndex(e3idx,1),
                cell.face_neighbors[4],
            ),
            cell.level+1,
            nothing
        ),
        # Child 2
        DGFQuadrilateralCell(
            cell.self_index,
            c2idx,
            2,
            (
                cell.face_neighbors[1],
                cell.face_neighbors[2],
                FaceIndex(e4idx,1),
                FaceIndex(e1idx,2),
            ),
            cell.level+1,
            nothing
        ),
        # Child 3
        DGFQuadrilateralCell(
            cell.self_index,
            c3idx,
            3,
            (
                FaceIndex(e1idx,4),
                cell.face_neighbors[2],
                cell.face_neighbors[3],
                FaceIndex(e1idx,2),
            ),
            cell.level+1,
            nothing
        ),
        # Child 4
        DGFQuadrilateralCell(
            cell.self_index,
            c3idx,
            3,
            (
                FaceIndex(e1idx,4),
                cell.face_neighbors[2],
                cell.face_neighbors[3],
                FaceIndex(e1idx,2),
            ),
            cell.level+1,
            nothing
        ),
    )
end
