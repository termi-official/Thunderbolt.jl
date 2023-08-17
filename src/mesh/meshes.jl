
# abstract type AbstractSurface end
# abstract type AbstractPath end
# struct ConformingFace3D <: AbstractSurface
#     cellidx_a::Int
#     cellidx_b::Int
#     defining_nodes::NTuple{3,Int}
# end

# struct SimpleEdge3D <: AbstractPath
#     defining_nodes::NTuple{3,Int}
# end

const LinearCellGeometry = Union{Hexahedron, Tetrahedron, Pyramid, Wedge, Triangle, Quadrilateral, Line}

"""
SimpleMesh3D{C <: AbstractCell, T <: Real}

A grid which also has information abouts its vertices, faces and edges.
"""
struct SimpleMesh3D{C <: AbstractCell, T <: Real} <: AbstractGrid{3}
    grid::Grid{3, C, T}
    mfaces::OrderedDict{NTuple{3,Int}, Int} # Maps "sortface"-representation to id
    medges::OrderedDict{NTuple{2,Int}, Int} # Maps "sortedge"-representation to id
    mvertices::OrderedDict{Int, Int} # Maps node to id
    number_of_cells_by_type::Dict{DataType, Int}
end

global_edges(mgrid::SimpleMesh3D, cell) = [mgrid.medges[sedge] for sedge ∈ first.(sortedge.(edges(cell)))]
global_faces(mgrid::SimpleMesh3D, cell) = [mgrid.mfaces[sface] for sface ∈ first.(sortface.(faces(cell)))]
global_vertices(mgrid::SimpleMesh3D, cell) = [mgrid.mvertices[v] for v ∈ vertices(cell)]

num_nodes(mgrid::SimpleMesh3D) = length(mgrid.grid.nodes)
num_faces(mgrid::SimpleMesh3D) = length(mgrid.mfaces)
num_edges(mgrid::SimpleMesh3D) = length(mgrid.medges)
num_vertices(mgrid::SimpleMesh3D) = length(mgrid.mvertices)

function SimpleMesh3D(grid::Grid{3,C,T}) where {C, T}
    mfaces = OrderedDict{NTuple{3,Int}, Int}()
    medges = OrderedDict{NTuple{2,Int}, Int}()
    mvertices = OrderedDict{Int, Int}()
    next_face_idx = 1
    next_edge_idx = 1
    next_vertex_idx = 1
    number_of_cells_by_type = Dict{DataType, Int}()
    for cell ∈ getcells(grid)
        cell_type = typeof(cell)
        if haskey(number_of_cells_by_type, cell_type)
            number_of_cells_by_type[cell_type] += 1
        else
            number_of_cells_by_type[cell_type] = 1
        end

        for v ∈ vertices(cell)
            if !haskey(mvertices, v)
                mvertices[v] = next_vertex_idx
                next_vertex_idx += 1
            end
        end
        for e ∈ first.(sortedge.(edges(cell)))
            if !haskey(medges, e)
                medges[e] = next_edge_idx
                next_edge_idx += 1
            end
        end
        for f ∈ first.(sortface.(faces(cell)))
            if !haskey(mfaces, f)
                mfaces[f] = next_face_idx
                next_face_idx += 1
            end
        end
    end
    return SimpleMesh3D(grid, mfaces, medges, mvertices, number_of_cells_by_type)
end
