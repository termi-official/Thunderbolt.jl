###################
# Device Iterator #
###################

abstract type AbstractDeviceCellIterator end


ncells(iterator::AbstractDeviceCellIterator) = iterator.n_cells ## any subtype has to have `n_cells` field


struct DeviceCellIterator{DH <: DeviceDofHandlerData, GRID <: DeviceGrid, Ti <: Integer, CellMem<: AbstractCellMem} <: AbstractDeviceCellIterator
    dh::DH
    grid::GRID
    n_cells::Ti # depends whether we are iterating over all cells (i.e. all the dh) or a subset of cells (i.e. subdh)
    cell_mem::CellMem
    sdh_idx::Ti # subdomain index (this is similar to set on the CPU side, but instead of set we have subdomain index (-1 -> all cells), otherwise index of the subdomain)
end

struct DeviceOutOfBoundCellIterator <: AbstractDeviceCellIterator end  # used to handle the case for out of bound threads



#####################
# Device Cell Cache #
#####################

abstract type AbstractDeviceCellCache  end

struct DeviceCellCache{Ti <: Integer, DOFS <: AbstractVector{Ti}, NN, NODES <: SVector{NN, Ti}, X, COORDS <: SVector{X}, CellMem<: AbstractCellMem} <: AbstractDeviceCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Ti
    nodes::NODES
    cell_mem::CellMem
end



@inline function cellke(::AbstractDeviceCellCache)
    throw(ArgumentError("cellke should be implemented in the derived type"))
end

@inline function cellfe(::AbstractDeviceCellCache)
    throw(ArgumentError("cellfe should be implemented in the derived type"))
end

# accessors
Ferrite.getnodes(cc::DeviceCellCache) = cc.nodes
Ferrite.getcoordinates(cc::DeviceCellCache) = cc.coords
Ferrite.celldofs(cc::DeviceCellCache) = cc.dofs
Ferrite.cellid(cc::DeviceCellCache) = cc.cellid