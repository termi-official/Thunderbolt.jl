# Utility which holds partial information for assembly.
struct GPUGrid{sdim, C<:AbstractCell, T<:Real, CellDataType <: AbstractGPUVector{C}, NodeDataType <: AbstractGPUVector} <: AbstractGrid{sdim}
    cells::CellDataType
    nodes::NodeDataType
    #TODO subdomain info
end
Adapt.@adapt_structure GPUGrid

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

function Adapt.adapt_structure(to, grid::SimpleMesh)
    return adapt_structure(to, grid.grid)
end
