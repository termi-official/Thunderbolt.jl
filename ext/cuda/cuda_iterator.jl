

_cellmem(buffer_alloc::AbstractDeviceSharedMemAlloc, ::Integer) = error("please provide concrete implementation for $(typeof(buffer_alloc))."))


function _cellmem(buffer_alloc::GlobalMemAlloc, i::Integer)
    ke = cellke(buffer_alloc, i)
    fe = cellfe(buffer_alloc, i)
    return FullCellMem(ke, fe)
end
function _cellmem(buffer_alloc::GlobalRHSMemAlloc, i::Integer)
    fe = cellfe(buffer_alloc, i)
    return RHSCellMem(fe)
end

_cellmem(buffer_alloc::AbstractDeviceGlobalMemAlloc, ::Integer) = error("please provide concrete implementation for $(typeof(buffer_alloc)).")

function _cellmem(buffer_alloc::SharedMemAlloc, i::Integer)
    block_ke = buffer_alloc.Ke()
    block_fe = buffer_alloc.fe()
    ke = @view block_ke[i, :, :]
    fe = @view block_fe[i, :]
    return FullCellMem(ke, fe)
end
function _cellmem(buffer_alloc::SharedRHSMemAlloc, i::Integer)
    block_fe = buffer_alloc.fe()
    fe = @view block_fe[i, :]
    return RHSCellMem(fe)
end

function _cell_iterator(dh::DeviceDofHandlerData,sdh_idx::Integer,n_cells::Integer)
    grid = dh.grid
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return DeviceOutOfBoundCellIterator()
    cell_mem = NoCellMem()
    return DeviceCellIterator(dh, grid, n_cells, cell_mem ,sdh_idx)
end

function _cell_iterator(dh::DeviceDofHandlerData,sdh_idx::Integer,n_cells::Integer, buffer_alloc::AbstractDeviceGlobalMemAlloc)
    grid = get_grid(dh)
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return DeviceOutOfBoundCellIterator()
    cell_mem = _cellmem(buffer_alloc, global_thread_id)
    return DeviceCellIterator(dh, grid, n_cells, cell_mem,sdh_idx)
end

function _cell_iterator(dh::DeviceDofHandlerData,sdh_idx::Integer,n_cells::Integer, buffer_alloc::AbstractDeviceSharedMemAlloc)
    grid = Ferrite.get_grid(dh)
    local_thread_id = threadIdx().x
    cell_mem = _cellmem(buffer_alloc, local_thread_id)
    return DeviceCellIterator(dh, grid, n_cells, cell_mem,sdh_idx)
end

 
Ferrite.CellIterator(dh::DeviceDofHandlerData, buffer_alloc::AbstractDeviceGlobalMemAlloc) = _cell_iterator(dh, -1,  dh |> get_grid |> getncells |> Int32, buffer_alloc) ## iterate over all cells

function Ferrite.CellIterator(dh::DeviceDofHandlerData,sdh_idx::Ti, buffer_alloc::AbstractDeviceGlobalMemAlloc) where {Ti <: Integer}
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
    n_cells = dh.subdofhandlers[sdh_idx].cellset |> length |> (x -> convert(Ti, x)) 
    return _cell_iterator(dh, sdh_idx,n_cells)
end

Ferrite.CellIterator(dh::DeviceDofHandlerData, buffer_alloc::AbstractDeviceSharedMemAlloc) = _cell_iterator(dh, -1,dh |> get_grid |> getncells |> Int32, buffer_alloc) ## iterate over all cells
   

function Ferrite.CellIterator(dh::DeviceDofHandlerData,sdh_idx::Integer, buffer_alloc::AbstractDeviceSharedMemAlloc)
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
    dh = iterator.dh
    grid = iterator.grid
    sdh_idx = iterator.sdh_idx
    cellid = e
    sdh_idx <= 0 || (cellid = dh.subdofhandlers[sdh_idx].cellset[e])
    cell = Ferrite.getcells(grid, e)

    # Extract the node IDs of the cell.
    nodes = SVector(convert.(Ti, Ferrite.get_node_ids(cell))...)

    # Extract the degrees of freedom for the cell.
    dofs = Ferrite.celldofs(dh, e)

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
KeTrait(::Type{<:RHSCellMem}) = HasNoKe()
FeTrait(::Type{<:AbstractCellMem}) = HasFe()
FeTrait(::Type{<:JacobianCellMem}) = HasNoFe()

@inline function _cellke(::HasKe, cc::DeviceCellCache)
    ke =  cc.cell_mem.ke
    return CUDA.fill!(ke, 0.0f0)
end

_cellke(::HasNoKe, ::DeviceCellCache) = error("$(typeof(cc.cell_mem)) does not have ke field.")

cellke(cc::DeviceCellCache) = _cellke(KeTrait(typeof(cc.cell_mem)), cc)

@inline function _cellfe(::HasFe, cc::DeviceCellCache)
    fe =  cc.cell_mem.fe
    return CUDA.fill!(fe, 0.0f0)
end

_cellfe(::HasNoFe, ::DeviceCellCache) = error("$(typeof(cc.cell_mem)) does not have fe field.")

cellfe(cc::DeviceCellCache) = _cellfe(FeTrait(typeof(cc.cell_mem)), cc)