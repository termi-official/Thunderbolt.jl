####################################
# mem_alloc.jl & cuda_mem_alloc.jl #
####################################

abstract type AbstractMemAllocObjects{Tv <: Real} end
struct RHSObject{Tv<:Real} <:AbstractMemAllocObjects{Tv}  end
struct JacobianObject{Tv<:Real} <:AbstractMemAllocObjects{Tv}  end
struct FullObject{Tv<:Real} <:AbstractMemAllocObjects{Tv}  end


abstract type AbstractMemAlloc end

struct DynamicSharedMemFunction{N, Tv <: Real, Ti <: Integer}
    mem_size::NTuple{N, Ti}
    offset::Ti
end


function (dsf::DynamicSharedMemFunction{N, Tv, Ti})() where {N, Tv, Ti}
    mem_size = dsf.mem_size
    offset = dsf.offset
    return @cuDynamicSharedMem(Tv, mem_size, offset)
end

abstract type AbstractCudaMemAlloc <: AbstractMemAlloc end
abstract type AbstractSharedMemAlloc <: AbstractCudaMemAlloc end
abstract type AbstractGlobalMemAlloc <: AbstractCudaMemAlloc end

struct SharedMemAlloc{N, M, Tv <: Real, Ti <: Integer} <: AbstractSharedMemAlloc
    Ke::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    fe::DynamicSharedMemFunction{M, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

mem_size(alloc::AbstractSharedMemAlloc) = alloc.tot_mem_size


function _can_use_dynshmem(required_shmem::Integer)
    dev = device()
    MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return required_shmem < MAX_DYN_SHMEM
end


function try_allocate_shared_mem(::Type{FullObject{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * (n_basefuncs) * n_basefuncs + sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    Ke = DynamicSharedMemFunction{3, Tv, Ti}((block_dim, n_basefuncs, n_basefuncs), convert(Ti, 0))
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, sizeof(Tv) * block_dim * n_basefuncs * n_basefuncs))
    return SharedMemAlloc(Ke, fe, shared_mem)
end


struct GlobalMemAlloc{LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractGlobalMemAlloc
    Kes::LOCAL_MATRICES ## global level allocation (i.e. memory for all blocks -> 3rd order tensor)
    fes::LOCAL_VECTORS  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

cellke(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.Kes[e, :, :]
cellfe(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.fes[e, :]


function allocate_global_mem(::Type{FullObject{Tv}}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    Kes = CUDA.zeros(Tv, n_cells, n_basefuncs, n_basefuncs)
    fes = CUDA.zeros(Tv, n_cells, n_basefuncs)
    return GlobalMemAlloc(Kes, fes)
end


struct SharedRHSMemAlloc{N, Tv <: Real, Ti <: Integer} <: AbstractSharedMemAlloc
    fe::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

function try_allocate_shared_mem(::Type{RHSObject{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, sizeof(Tv) * block_dim * n_basefuncs * n_basefuncs))
    return SharedRHSMemAlloc( fe, shared_mem)
end

function try_allocate_shared_mem(::Type{JacobianObject{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    throw(ErrorException("Not implemented"))
end


struct GlobalRHSMemAlloc{LOCAL_VECTORS} <: AbstractGlobalMemAlloc
    fes::LOCAL_VECTORS  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

cellfe(alloc::GlobalRHSMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.fes[e, :]


function allocate_global_mem(::Type{RHSObject{Tv}}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    fes = CUDA.zeros(Tv, n_cells, n_basefuncs)
    return GlobalRHSMemAlloc( fes)
end


####################
# GPUDofHandler.jl #
####################


# This file defines the `GPUDofHandler` type, which is a degree of freedom handler that is stored on the GPU.
# Therefore most of the functions are same as the ones defined in dof_handler.jl, but executable on the GPU.

"""
    AbstractGPUDofHandler <: Ferrite.AbstractDofHandler

Abstract type representing degree-of-freedom (DoF) handlers for GPU-based
finite element computations. This serves as the base type for GPU-specific
DoF handler implementations.
"""
abstract type AbstractGPUDofHandler <: AbstractDofHandler end

# TODO: add subdofhandlers
struct GPUDofHandler{CDOFS <: AbstractArray{<:Number, 1}, VEC_INT <: AbstractArray{Int32, 1}, GRID <: AbstractGrid} <: AbstractGPUDofHandler
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    ndofs_cell::VEC_INT
end


ndofs_per_cell(dh::GPUDofHandler, i::Int32) = dh.ndofs_cell[i]


cell_dof_offset(dh::GPUDofHandler, i::Int32) = dh.cell_dofs_offset[i]


get_grid(dh::GPUDofHandler) = dh.grid

function celldofs(dh::GPUDofHandler, i::Int32)
    offset = cell_dof_offset(dh, i)
    ndofs = ndofs_per_cell(dh, i)
    view = @view dh.cell_dofs[offset:(offset + ndofs - Int32(1))]
    return view
end


####################
# cuda_iterator.jl #
####################

##### GPUCellIterator #####


abstract type AbstractCellMem end
struct RHSCellMem{VectorType} <: AbstractCellMem 
    fe::VectorType
end

struct JacobianCellMem{MatrixType} <: AbstractCellMem 
    ke::MatrixType
end

struct FullCellMem{MatrixType, VectorType} <: AbstractCellMem 
    ke::MatrixType
    fe::VectorType
end

"""
    AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator

Abstract type representing CUDA cell iterators for finite element computations
on the GPU. It provides the base for implementing multiple cell iteration strategies (e.g. with shared memory, with global memory).
"""
abstract type AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator end


ncells(iterator::AbstractCUDACellIterator) = iterator.n_cells ## any subtype has to have `n_cells` field


struct CUDACellIterator{DH <: Ferrite.GPUDofHandler, GRID <: Ferrite.AbstractGPUGrid, Ti <: Integer, CellMem<: AbstractCellMem} <: AbstractCUDACellIterator
    dh::DH
    grid::GRID
    n_cells::Ti
    cell_mem::CellMem
end

struct CudaOutOfBoundCellIterator <: AbstractCUDACellIterator end  # used to handle the case for out of bound threads (global memory only)

_cellmem(buffer_alloc::AbstractSharedMemAlloc, ::Integer) = throw(ErrorException("please provide concrete implementation for $(typeof(buffer_alloc))."))

function _cellmem(buffer_alloc::SharedMemAlloc, i::Integer)
    ke = cellke(buffer_alloc, i)
    fe = cellfe(buffer_alloc, i)
    return FullCellMem(ke, fe)
end
function _cellmem(buffer_alloc::SharedRHSMemAlloc, i::Integer)
    fe = cellfe(buffer_alloc, i)
    return RHSCellMem(fe)
end


function Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, buffer_alloc::AbstractGlobalMemAlloc)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return CudaOutOfBoundCellIterator()
    cell_mem = _cellmem(buffer_alloc, global_thread_id)
    return CUDACellIterator(dh, grid, n_cells, cell_mem)
end



_cellmem(buffer_alloc::AbstractGlobalMemAlloc, ::Integer) = throw(ErrorException("please provide concrete implementation for $(typeof(buffer_alloc))."))

function _cellmem(buffer_alloc::GlobalMemAlloc, i::Integer)
    block_ke = buffer_alloc.Ke()
    block_fe = buffer_alloc.fe()
    ke = @view block_ke[i, :, :]
    fe = @view block_fe[i, :]
    return FullCellMem(ke, fe)
end
function _cellmem(buffer_alloc::GlobalRHSMemAlloc, i::Integer)
    block_fe = buffer_alloc.fe()
    fe = @view block_fe[i, :]
    return RHSCellMem(fe)
end



function Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, buffer_alloc::AbstractSharedMemAlloc)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    local_thread_id = threadIdx().x
    cell_mem = _cellmem(buffer_alloc, local_thread_id)
    return CUDACellIterator(dh, grid, n_cells, cell_mem)
end


function Base.iterate(iterator::CUDACellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= ncells(iterator)  || return nothing
    return (_makecache(iterator, i), i)
end


function Base.iterate(iterator::CUDACellIterator, state)
    stride = blockDim().x * gridDim().x
    i = state + stride
    i <= ncells(iterator) || return nothing
    return (_makecache(iterator, i), i)
end


Base.iterate(::CudaOutOfBoundCellIterator) = nothing

Base.iterate(::CudaOutOfBoundCellIterator, state) = nothing # I believe this is not necessary


struct GPUCellCache{Ti <: Integer, DOFS <: AbstractVector{Ti}, NN, NODES <: SVector{NN, Ti}, X, COORDS <: SVector{X}, CellMem<: AbstractCellMem} <: Ferrite.AbstractKernelCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Ti
    nodes::NODES
    cell_mem::CellMem
end


function _makecache(iterator::AbstractCUDACellIterator, e::Integer)
    dh = iterator.dh
    grid = iterator.grid
    cellid = e
    cell = Ferrite.getcells(grid, e)

    # Extract the node IDs of the cell.
    nodes = SVector(convert.(Int32, Ferrite.get_node_ids(cell))...)

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

    # Return the GPUCellCache containing the cell's data.
    return GPUCellCache(coords, dofs, cellid, nodes, iterator.cell_mem)
end


Ferrite.getnodes(cc::GPUCellCache) = cc.nodes
Ferrite.getcoordinates(cc::GPUCellCache) = cc.coords
Ferrite.celldofs(cc::GPUCellCache) = cc.dofs
Ferrite.cellid(cc::GPUCellCache) = cc.cellid


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

@inline function _cellke(::HasKe, cc::GPUCellCache)
    ke =  cc.cell_mem.ke
    return CUDA.fill!(ke, 0.0f0)
end

_cellke(::HasNoKe, ::GPUCellCache) = throw(ErrorException("$(typeof(cc.cell_mem)) does not have ke field."))

Ferrite.cellke(cc::GPUCellCache) = _cellke(KeTrait(typeof(cc.cell_mem)), cc)

@inline function _cellfe(::HasFe, cc::GPUCellCache)
    fe =  cc.cell_mem.fe
    return CUDA.fill!(fe, 0.0f0)
end

_cellfe(::HasNoFe, ::GPUCellCache) = throw(ErrorException("$(typeof(cc.cell_mem)) does not have fe field."))

Ferrite.cellfe(cc::GPUCellCache) = _cellfe(FeTrait(typeof(cc.cell_mem)), cc)