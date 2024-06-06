
"""
SimpleMesh{C <: AbstractCell, T <: Real} <: AbstractGrid{3}

A grid which also has information abouts its vertices, faces and edges.
"""
struct SimpleMesh{sdim, C <: AbstractCell, T <: Real} <: AbstractGrid{sdim}
    grid::Grid{sdim, C, T}
    mfaces::OrderedDict{NTuple{3,Int}, Int} # Maps "sortface"-representation to id
    medges::OrderedDict{NTuple{2,Int}, Int} # Maps "sortedge"-representation to id
    mvertices::OrderedDict{Int, Int} # Maps node to id
    number_of_cells_by_type::Dict{DataType, Int}
end

global_edges(mgrid::SimpleMesh, cell) = [mgrid.medges[sedge] for sedge ∈ first.(sortedge.(edges(cell)))]
global_faces(mgrid::SimpleMesh, cell) = [mgrid.mfaces[sface] for sface ∈ first.(sortface.(faces(cell)))]
global_vertices(mgrid::SimpleMesh, cell) = [mgrid.mvertices[v] for v ∈ vertices(cell)]

num_nodes(mgrid::SimpleMesh) = length(mgrid.grid.nodes)
num_faces(mgrid::SimpleMesh) = length(mgrid.mfaces)
num_edges(mgrid::SimpleMesh) = length(mgrid.medges)
num_vertices(mgrid::SimpleMesh) = length(mgrid.mvertices)

elementtypes(::SimpleMesh{sdim,Triangle}) where sdim = @SVector [Triangle]
elementtypes(::SimpleMesh{sdim,Quadrilateral}) where sdim = @SVector [Quadrilateral]
elementtypes(::SimpleMesh{3,Tetrahedron}) = @SVector [Tetrahedron]
elementtypes(::SimpleMesh{3,Hexahedron}) = @SVector [Hexahedron]

function to_mesh(grid::Grid)
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
    return SimpleMesh(grid, mfaces, medges, mvertices, number_of_cells_by_type)
end

# Ferrite compat layer for the mesh
@inline Ferrite.getncells(mesh::SimpleMesh) = Ferrite.getncells(mesh.grid)
@inline Ferrite.getcells(mesh::SimpleMesh) = Ferrite.getcells(mesh.grid)
@inline Ferrite.getcells(mesh::SimpleMesh, v::Union{Int, Vector{Int}}) = Ferrite.getcells(mesh.grid, v)
@inline Ferrite.getcells(mesh::SimpleMesh, setname::String) = Ferrite.getcells(mesh.grid, setname)
@inline Ferrite.getcelltype(mesh::SimpleMesh) = Ferrite.getcelltype(mesh.grid)
@inline Ferrite.getcelltype(mesh::SimpleMesh, i::Int) = Ferrite.getcelltype(mesh.grid)

@inline Ferrite.getnnodes(mesh::SimpleMesh) = Ferrite.getnnodes(mesh.grid)
@inline Ferrite.nnodes_per_cell(mesh::SimpleMesh, i::Int=1) = Ferrite.nnodes_per_cell(mesh.grid, i)
@inline Ferrite.getnodes(mesh::SimpleMesh) = Ferrite.getnodes(mesh.grid)
@inline Ferrite.getnodes(mesh::SimpleMesh, v::Union{Int, Vector{Int}}) = Ferrite.getnodes(mesh.grid, v)
@inline Ferrite.getnodes(mesh::SimpleMesh, setname::String) = Ferrite.getnodes(mesh.grid, setname)

@inline Ferrite.vtk_grid(filename::String, mesh::SimpleMesh; kwargs...) = Ferrite.vtk_grid(filename, mesh.grid, kwargs...)

@inline Ferrite.get_coordinate_type(::SimpleMesh{sdim, <:Any, T}) where {sdim,T} = Vec{sdim,T} 

@inline Ferrite.CellIterator(mesh::SimpleMesh) = CellIterator(mesh.grid)
