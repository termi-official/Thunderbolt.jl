
# This file contains the cuda implementation for `device_iterator.jl` in the `ferrite-addons` module.
# this iterator can iterates over all cells in the grid or in a subdomain depends on the subdomain index (sdh_idx = -1 -> all cells, otherwise index of the subdomain)
# it takes in its constructor the `mem_alloc` which is the memory allocation for the entire group and 
# from it extracts the `cell_mem` which is the memory allocation for each cell (i.e. memory per cell)

# iterator with no memory allocation (for testing purposes)
function _cell_iterator(dh::DeviceDofHandlerData,sdh_idx::Integer,n_cells::Integer)
    grid = dh.grid
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return DeviceOutOfBoundCellIterator()
    cell_mem = NoCellMem()
    return  DeviceCellIterator(dh, grid, n_cells, cell_mem ,sdh_idx)
end

# iterator with global memory allocation
function _cell_iterator(dh::DeviceDofHandlerData,sdh_idx::Integer,n_cells::Integer, global_mem::AbstractDeviceGlobalMem)
    grid = get_grid(dh)
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id 
    global_thread_id <= n_cells || return DeviceOutOfBoundCellIterator()
    cell_mem = cellmem(global_mem, global_thread_id)
    return DeviceCellIterator(dh, grid, n_cells, cell_mem,sdh_idx)
end

# iterator with shared memory allocation
function _cell_iterator(dh::DeviceDofHandlerData,sdh_idx::Integer,n_cells::Integer, buffer_alloc::AbstractDeviceSharedMem)
    grid = Ferrite.get_grid(dh)
    local_thread_id = threadIdx().x
    cell_mem = cellmem(buffer_alloc, local_thread_id)
    return DeviceCellIterator(dh, grid, n_cells, cell_mem,sdh_idx)
end

# iterator with global memory allocation over the whole grid
Ferrite.CellIterator(dh::DeviceDofHandlerData, buffer_alloc::AbstractDeviceGlobalMem) = _cell_iterator(dh, -1,  dh |> get_grid |> getncells |> Int32, buffer_alloc) ## iterate over all cells
# iterator with shared memory allocation over the whole grid
Ferrite.CellIterator(dh::DeviceDofHandlerData, buffer_alloc::AbstractDeviceSharedMem) = _cell_iterator(dh, -1,dh |> get_grid |> getncells |> Int32, buffer_alloc) ## iterate over all cells

function Ferrite.CellIterator(dh::DeviceDofHandlerData,sdh_idx::Ti, buffer_alloc::AbstractDeviceGlobalMem) where {Ti <: Integer}
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    sdh_idx ∉ 1:length(dh.subdofhandlers) && return DeviceOutOfBoundCellIterator()
    n_cells = dh.subdofhandlers[sdh_idx].cellset |> length |> (x -> convert(Ti, x)) 
    return _cell_iterator(dh, sdh_idx,n_cells,buffer_alloc)
end


function Ferrite.CellIterator(dh::DeviceDofHandlerData,sdh_idx::Ti) where {Ti <: Integer}
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    sdh_idx ∉ 1:length(dh.subdofhandlers) && return DeviceOutOfBoundCellIterator()
    n_cells =dh.subdofhandlers[sdh_idx].cellset |> length |> (x -> convert(Ti, x)) 
    return _cell_iterator(dh, sdh_idx,n_cells)
end


function Ferrite.CellIterator(dh::DeviceDofHandlerData,sdh_idx::Integer, buffer_alloc::AbstractDeviceSharedMem)
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    sdh_idx ∉ 1:length(dh.subdofhandlers) && return DeviceOutOfBoundCellIterator()
    n_cells = dh.subdofhandlers[sdh_idx].cellset |> length |> (x -> convert(typeof(dh.ndofs), x))  
    return _cell_iterator(dh, sdh_idx,n_cells, buffer_alloc)
end
   

function Base.iterate(iterator::DeviceCellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= ncells(iterator)  || return nothing
    return (_makecache(iterator, i), i)
end


function Base.iterate(iterator::DeviceCellIterator, state)
    stride = blockDim().x * gridDim().x
    i = state + stride
    i <= ncells(iterator) || return nothing
    return (_makecache(iterator, i), i)
end


Base.iterate(::DeviceOutOfBoundCellIterator) = nothing

Base.iterate(::DeviceOutOfBoundCellIterator, state) = nothing # I believe this is not necessary



# cuda cell cache #
function _makecache(iterator::AbstractDeviceCellIterator, e::Ti) where {Ti <: Integer}
    # NOTE: e is the global thread index, so in case of iterating over the whole cells in the grid, e is the cell index
    # whereas in the case of we are only iterating over a subdomain, e is just the iterator index from 1:n_cells in subdomain
    dh = iterator.dh
    grid = iterator.grid
    sdh_idx = iterator.sdh_idx
    cellid = e
    sdh_idx <= 0 || (cellid = dh.subdofhandlers[sdh_idx].cellset[e])
    cell = Ferrite.getcells(grid, cellid)

    #Extract the node IDs of the cell.
    nodes = SVector(convert.(Ti, Ferrite.get_node_ids(cell))...)

    # Extract the degrees of freedom for the cell.
    dofs = Ferrite.celldofs(dh, cellid)

    # Get the coordinates of the nodes of the cell.
    CT = Ferrite.get_coordinate_type(grid)
    N = Ferrite.nnodes(cell)
    x = MVector{N, CT}(undef)
    for i in eachindex(x)
        x[i] = Ferrite.get_node_coordinate(grid, nodes[i])
    end
    coords = SVector(x...)

    # Return the DeviceCellCache containing the cell's data.
    return DeviceCellCache(coords, dofs, cellid, nodes, iterator.cell_mem)
end


abstract type AbstractKeTrait end
struct HasKe <: AbstractKeTrait end
struct HasNoKe <: AbstractKeTrait end
abstract type AbstractFeTrait end
struct HasFe <: AbstractFeTrait end
struct HasNoFe <: AbstractFeTrait end

KeTrait(::Type{<:AbstractCellMem}) = HasKe()
KeTrait(::Type{<:FeMemShape}) = HasNoKe()
FeTrait(::Type{<:AbstractCellMem}) = HasFe()
FeTrait(::Type{<:KeMemShape}) = HasNoFe()

@inline function _cellke(::HasKe, cc::DeviceCellCache)
    ke =  cc.cell_mem.ke
    FT = eltype(ke)
    return CUDA.fill!(ke, zero(FT))
end

_cellke(::HasNoKe, ::DeviceCellCache) = error("$(typeof(cc.cell_mem)) does not have ke field.")

Thunderbolt.FerriteUtils.cellke(cc::DeviceCellCache) = _cellke(KeTrait(typeof(cc.cell_mem)), cc)

@inline function _cellfe(::HasFe, cc::DeviceCellCache)
    fe =  cc.cell_mem.fe
    FT = eltype(fe)
    return CUDA.fill!(fe, zero(FT))
end

_cellfe(::HasNoFe, ::DeviceCellCache) = error("$(typeof(cc.cell_mem)) does not have fe field.")

Thunderbolt.FerriteUtils.cellfe(cc::DeviceCellCache) = _cellfe(FeTrait(typeof(cc.cell_mem)), cc)