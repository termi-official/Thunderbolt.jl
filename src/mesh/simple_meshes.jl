
"""
SimpleMesh3D{C <: AbstractCell, T <: Real} <: AbstractGrid{3}

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

elementtypes(::SimpleMesh3D{Tetrahedron}) = @SVector [Tetrahedron]
elementtypes(::SimpleMesh3D{Hexahedron}) = @SVector [Hexahedron]

function to_mesh(grid::Grid{3,C,T}) where {C, T}
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


"""
SimpleMesh2D{C <: AbstractCell, T <: Real} <: AbstractGrid{2}

A grid which also has information abouts its vertices, faces and edges.
"""
struct SimpleMesh2D{C <: AbstractCell, T <: Real} <: AbstractGrid{2}
grid::Grid{2, C, T}
mfaces::OrderedDict{NTuple{2,Int}, Int} # Maps "sortface"-representation to id
mvertices::OrderedDict{Int, Int} # Maps node to id
number_of_cells_by_type::Dict{DataType, Int}
end

function to_mesh(grid::Grid{2,C,T}) where {C, T}
mfaces = OrderedDict{NTuple{2,Int}, Int}()
mvertices = OrderedDict{Int, Int}()
next_face_idx = 1
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
    for f ∈ first.(sortface.(faces(cell)))
        if !haskey(mfaces, f)
            mfaces[f] = next_face_idx
            next_face_idx += 1
        end
    end
end
return SimpleMesh2D(grid, mfaces, mvertices, number_of_cells_by_type)
end


global_faces(mgrid::SimpleMesh2D, cell) = [mgrid.mfaces[sface] for sface ∈ first.(sortface.(faces(cell)))]
global_vertices(mgrid::SimpleMesh2D, cell) = [mgrid.mvertices[v] for v ∈ vertices(cell)]

num_nodes(mgrid::SimpleMesh2D) = length(mgrid.grid.nodes)
num_faces(mgrid::SimpleMesh2D) = length(mgrid.mfaces)
num_vertices(mgrid::SimpleMesh2D) = length(mgrid.mvertices)

elementtypes(::SimpleMesh2D{Triangle}) = @SVector [Triangle]
elementtypes(::SimpleMesh2D{Quadrilateral}) = @SVector [Quadrilateral]

# Ferrite compat layer for the mesh
@inline Ferrite.getncells(mesh::Union{SimpleMesh2D,SimpleMesh3D}) = Ferrite.getncells(mesh.grid)
@inline Ferrite.getcells(mesh::Union{SimpleMesh2D,SimpleMesh3D}) = Ferrite.getcells(mesh.grid)
@inline Ferrite.getcells(mesh::Union{SimpleMesh2D,SimpleMesh3D}, v::Union{Int, Vector{Int}}) = Ferrite.getcells(mesh.grid, v)
@inline Ferrite.getcells(mesh::Union{SimpleMesh2D,SimpleMesh3D}, setname::String) = Ferrite.getcells(mesh.grid, setname)
@inline Ferrite.getcelltype(mesh::Union{SimpleMesh2D,SimpleMesh3D}) = Ferrite.getcelltype(mesh.grid)
@inline Ferrite.getcelltype(mesh::Union{SimpleMesh2D,SimpleMesh3D}, i::Int) = Ferrite.getcelltype(mesh.grid)

@inline Ferrite.getnnodes(mesh::Union{SimpleMesh2D,SimpleMesh3D}) = Ferrite.getnnodes(mesh.grid)
@inline Ferrite.nnodes_per_cell(mesh::Union{SimpleMesh2D,SimpleMesh3D}, i::Int=1) = Ferrite.nnodes_per_cell(mesh.grid, i)
@inline Ferrite.getnodes(mesh::Union{SimpleMesh2D,SimpleMesh3D}) = Ferrite.getnodes(mesh.grid)
@inline Ferrite.getnodes(mesh::Union{SimpleMesh2D,SimpleMesh3D}, v::Union{Int, Vector{Int}}) = Ferrite.getnodes(mesh.grid, v)
@inline Ferrite.getnodes(mesh::Union{SimpleMesh2D,SimpleMesh3D}, setname::String) = Ferrite.getnodes(mesh.grid, setname)

@inline Ferrite.vtk_grid(filename::AbstractString, mesh::Union{SimpleMesh2D,SimpleMesh3D}; kwargs...) = Ferrite.vtk_grid(filename, mesh.grid, kwargs...)

@inline Ferrite.get_coordinate_type(::SimpleMesh2D{C,T}) where {C,T} = Vec{2,T} 
@inline Ferrite.get_coordinate_type(::SimpleMesh3D{C,T}) where {C,T} = Vec{3,T} 

@inline CellIterator(mesh::Union{SimpleMesh2D,SimpleMesh3D}) = CellIterator(mesh.grid)
