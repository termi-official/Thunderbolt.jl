# Utility which holds partial information for assembly.
struct GPUGrid{sdim, C<:Ferrite.AbstractCell, T<:Real, CellDataType <: AbstractVector{C}, NodeDataType <: AbstractVector} <: Ferrite.AbstractGrid{sdim}
    cells::CellDataType
    nodes::NodeDataType
    #TODO subdomain info
end

function Base.show(io::IO, mime::MIME"text/plain", data::GPUGrid)
    _show(io, mime, data, 0)
end

function _show(io::IO, mime::MIME"text/plain", grid::GPUGrid{sdim, C, T}, indent) where {sdim, C, T}
    offset = " "^indent
    print(io, offset, "GPUGrid{sdim=", sdim, ", T=", T, "}")
    if isconcretetype(eltype(grid.cells))
        typestrs = [repr(eltype(grid.cells))]
    else
        typestrs = sort!(repr.(OrderedSet(typeof(x) for x in grid.cells)))
    end
    print(io, " with ")
    join(io, typestrs, '/')
    println(io, " cells and $(length(grid.nodes)) nodes")
end

# commented out because SimpleMesh is not defined in FerriteUtils module.
# function Adapt.adapt_structure(to, grid::SimpleMesh)
#     return adapt_structure(to, grid.grid)
# end


Ferrite.get_coordinate_type(::GPUGrid{sdim, <:Any, T,<:Any,<:Any}) where {sdim, T} = Vec{sdim, T} # Node is baked into the mesh type.

@inline Ferrite.getcells(grid::GPUGrid, v::Ti) where {Ti <: Integer} = grid.cells[v]
@inline Ferrite.getnodes(grid::GPUGrid, v::Ti) where {Ti<: Integer} = grid.nodes[v]


function Ferrite.getcoordinates(grid::GPUGrid, e::Ti) where {Ti<: Integer}
    # e is the element index.
    CT = Ferrite.get_coordinate_type(grid)
    cell = getcells(grid, e)
    N = nnodes(cell)
    x = MVector{N, CT}(undef) # local array to store the coordinates of the nodes of the cell.
    node_ids = get_node_ids(cell)
    for i in 1:length(x)
        x[i] = get_node_coordinate(grid, node_ids[i])
    end

    return SVector(x...)
end