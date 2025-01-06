using CUDA
using Adapt

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
    return CUDA.@cuDynamicSharedMem(Tv, mem_size, offset)
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
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, 0))
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

###############
# gpu_grid.jl #
###############
# This file defines the GPUGrid type, which is a grid that is stored on the GPU. Therefore most of the
# functions are same as the ones defined in grid.jl, but executable on the GPU.

abstract type AbstractGPUGrid{dim} <: Ferrite.AbstractGrid{dim} end

struct GPUGrid{dim, CELLVEC <: AbstractArray, NODEVEC <: AbstractArray} <: AbstractGPUGrid{dim}
    cells::CELLVEC
    nodes::NODEVEC
end

function GPUGrid(
        cells::CELLVEC,
        nodes::NODEVEC
    ) where {C <: Ferrite.AbstractCell, CELLVEC <: AbstractArray{C, 1}, NODEVEC <: AbstractArray{Node{dim, T}}} where {dim, T}
    return GPUGrid{dim, CELLVEC, NODEVEC}(cells, nodes)
end

Ferrite.get_coordinate_type(::GPUGrid{dim, CELLVEC, NODEVEC}) where
{C <: Ferrite.AbstractCell, CELLVEC <: AbstractArray{C, 1}, NODEVEC <: AbstractArray{Node{dim, T}}} where
{dim, T} = Vec{dim, T} # Node is baked into the mesh type.


# Note: For functions that takes blockIdx as an argument, we need to use Int32 explicitly,
# otherwise the compiler will not be able to infer the type of the argument and throw a dynamic function invokation error.
@inline Ferrite.getcells(grid::GPUGrid, v::Union{Int32, Vector{Int32}}) = grid.cells[v]
@inline Ferrite.getnodes(grid::GPUGrid, v::Int32) = grid.nodes[v]

"""
    getcoordinates(grid::Ferrite.GPUGrid,e::Int32)

Return the coordinates of the nodes of the element `e` in the `GPUGrid` `grid`.
"""
function Ferrite.getcoordinates(grid::GPUGrid, e::Int32)
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
abstract type AbstractGPUDofHandler <: Ferrite.AbstractDofHandler end

struct GPUSubDofHandler{VEC_INT, Ti, VEC_IP} <: AbstractGPUDofHandler
    cellset::VEC_INT
    field_names::VEC_INT # cannot use symbols in GPU
    field_interpolations::VEC_IP
    ndofs_per_cell::Ti
end

## IDEA: to have multiple interfaces for dofhandlers (e.g. one domain dofhandler, multiple subdomains)
struct GPUDofHandler{SUB_DOFS <: AbstractArray{<:AbstractGPUDofHandler, 1}, CDOFS <: AbstractArray{<:Number, 1}, VEC_INT <: AbstractArray{Int32, 1}, GRID <: Ferrite.AbstractGrid} <: AbstractGPUDofHandler
    subdofhandlers::SUB_DOFS
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    cell_to_subdofhandler::VEC_INT
end


function ndofs_per_cell(dh::GPUDofHandler, cell::Ti) where {Ti <: Integer}
    sdhidx = dh.cell_to_subdofhandler[cell]
    sdhidx ∉ 1:length(dh.subdofhandlers) && return 0 # Dof handler is just defined on a subdomain
    return ndofs_per_cell(dh.subdofhandlers[sdhidx])
end
ndofs_per_cell(sdh::GPUSubDofHandler) = sdh.ndofs_per_cell
cell_dof_offset(dh::GPUDofHandler, i::Int32) = dh.cell_dofs_offset[i]
get_grid(dh::GPUDofHandler) = dh.grid

function Ferrite.celldofs(dh::GPUDofHandler, i::Int32)
    offset = cell_dof_offset(dh, i)
    ndofs = ndofs_per_cell(dh, i)
    view = @view dh.cell_dofs[offset:(offset + ndofs - Int32(1))]
    return view
end


####################
# cuda_iterator.jl #
####################

##### GPUCellIterator #####
# abstract types and interfaces
abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractKernelCellCache <: AbstractCellCache end
abstract type AbstractKernelCellIterator <: AbstractIterator end

@inline function cellke(::AbstractKernelCellCache)
    throw(ArgumentError("cellke should be implemented in the derived type"))
end

@inline function cellfe(::AbstractKernelCellCache)
    throw(ArgumentError("cellfe should be implemented in the derived type"))
end

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


abstract type AbstractCUDACellIterator <: AbstractKernelCellIterator end


ncells(iterator::AbstractCUDACellIterator) = iterator.n_cells ## any subtype has to have `n_cells` field


struct CUDACellIterator{DH <: GPUDofHandler, GRID <: AbstractGPUGrid, Ti <: Integer, CellMem<: AbstractCellMem} <: AbstractCUDACellIterator
    dh::DH
    grid::GRID
    n_cells::Ti
    cell_mem::CellMem
    sdh_idx::Ti # subdomain index (this is similar to set on the CPU side, but instead of set we have subdomain index (-1 -> all cells), otherwise index of the subdomain)
end

struct CudaOutOfBoundCellIterator <: AbstractCUDACellIterator end  # used to handle the case for out of bound threads (global memory only)

_cellmem(buffer_alloc::AbstractSharedMemAlloc, ::Integer) = throw(ErrorException("please provide concrete implementation for $(typeof(buffer_alloc))."))


function _cellmem(buffer_alloc::GlobalMemAlloc, i::Integer)
    ke = cellke(buffer_alloc, i)
    fe = cellfe(buffer_alloc, i)
    return FullCellMem(ke, fe)
end
function _cellmem(buffer_alloc::GlobalRHSMemAlloc, i::Integer)
    fe = cellfe(buffer_alloc, i)
    return RHSCellMem(fe)
end

_cellmem(buffer_alloc::AbstractGlobalMemAlloc, ::Integer) = error("please provide concrete implementation for $(typeof(buffer_alloc)).")

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

function _cell_iterator(dh::GPUDofHandler,sdh_idx::Integer,n_cells::Integer, buffer_alloc::AbstractGlobalMemAlloc)
    grid = get_grid(dh)
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return CudaOutOfBoundCellIterator()
    cell_mem = _cellmem(buffer_alloc, global_thread_id)
    return CUDACellIterator(dh, grid, n_cells, cell_mem,sdh_idx)
end

function _cell_iterator(dh::GPUDofHandler,sdh_idx::Integer,n_cells::Integer, buffer_alloc::AbstractSharedMemAlloc)
    grid = get_grid(dh)
    local_thread_id = threadIdx().x
    cell_mem = _cellmem(buffer_alloc, local_thread_id)
    return CUDACellIterator(dh, grid, n_cells, cell_mem,sdh_idx)
end

 
Ferrite.CellIterator(dh::GPUDofHandler, buffer_alloc::AbstractGlobalMemAlloc) = _cell_iterator(dh, -1,  dh |> get_grid |> getncells |> Int32, buffer_alloc) ## iterate over all cells

function Ferrite.CellIterator(dh::GPUDofHandler,sdh_idx::Integer, buffer_alloc::AbstractGlobalMemAlloc)
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    sdh_idx ∉ 1:length(dh.subdofhandlers) && return CudaOutOfBoundCellIterator()
    n_cells = dh.subdofhandlers[sdh_idx].cellset |> length |> Int32 
    return _cell_iterator(dh, sdh_idx,n_cells, buffer_alloc)
end

Ferrite.CellIterator(dh::GPUDofHandler, buffer_alloc::AbstractSharedMemAlloc) = _cell_iterator(dh, -1,dh |> get_grid |> getncells |> Int32, buffer_alloc) ## iterate over all cells
   

function Ferrite.CellIterator(dh::GPUDofHandler,sdh_idx::Integer, buffer_alloc::AbstractSharedMemAlloc)
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    sdh_idx ∉ 1:length(dh.subdofhandlers) && return CudaOutOfBoundCellIterator()
    n_cells = dh.subdofhandlers[sdh_idx].cellset |> length |> Int32 
    return _cell_iterator(dh, sdh_idx,n_cells, buffer_alloc)
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


struct GPUCellCache{Ti <: Integer, DOFS <: AbstractVector{Ti}, NN, NODES <: SVector{NN, Ti}, X, COORDS <: SVector{X}, CellMem<: AbstractCellMem} <: AbstractKernelCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Ti
    nodes::NODES
    cell_mem::CellMem
end


function _makecache(iterator::AbstractCUDACellIterator, e::Integer)
    dh = iterator.dh
    grid = iterator.grid
    sdh_idx = iterator.sdh_idx
    cellid = e
    sdh_idx <= 0 || (cellid = dh.subdofhandlers[sdh_idx].cellset[e])
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

cellke(cc::GPUCellCache) = _cellke(KeTrait(typeof(cc.cell_mem)), cc)

@inline function _cellfe(::HasFe, cc::GPUCellCache)
    fe =  cc.cell_mem.fe
    return CUDA.fill!(fe, 0.0f0)
end

_cellfe(::HasNoFe, ::GPUCellCache) = throw(ErrorException("$(typeof(cc.cell_mem)) does not have fe field."))

cellfe(cc::GPUCellCache) = _cellfe(FeTrait(typeof(cc.cell_mem)), cc)


############
# adapt.jl #
############
Adapt.@adapt_structure GPUGrid
Adapt.@adapt_structure GPUDofHandler
Adapt.@adapt_structure GPUSubDofHandler
Adapt.@adapt_structure GlobalMemAlloc

function _adapt_args(args)
    return tuple(((_adapt(arg) for arg in args) |> collect)...)
end


function _adapt(kgpu::CUSPARSE.CuSparseMatrixCSC)
    # custom adaptation
    return Adapt.adapt_structure(CUSPARSE.CuSparseDeviceMatrixCSC, kgpu)
end


function _adapt(obj::Any)
    # fallback to the default implementation
    return Adapt.adapt_structure(CuArray, obj)
end

## Adapt GlobalMemAlloc
function Adapt.adapt_structure(to, mem_alloc::GlobalMemAlloc)
    kes = Adapt.adapt_structure(to, mem_alloc.Kes)
    fes = Adapt.adapt_structure(to, mem_alloc.fes)
    return GlobalMemAlloc(kes, fes)
end


function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, StaticInterpolationValues(cv.geo_mapping))
    weights = Adapt.adapt(to, ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv)))
    return Ferrite.StaticCellValues(fv, gm, weights)
end


function Adapt.adapt_structure(to, iter::QuadratureValuesIterator)
    cv = Adapt.adapt_structure(to, iter.v)
    cell_coords = Adapt.adapt_structure(to, iter.cell_coords)
    return QuadratureValuesIterator(cv, cell_coords)
end

function Adapt.adapt_structure(to, qv::StaticQuadratureValues)
    det = Adapt.adapt_structure(to, qv.detJdV)
    N = Adapt.adapt_structure(to, qv.N)
    dNdx = Adapt.adapt_structure(to, qv.dNdx)
    M = Adapt.adapt_structure(to, qv.M)
    return StaticQuadratureValues(det, N, dNdx, M)
end
# function Adapt.adapt_structure(to, qv::StaticQuadratureView)
#     mapping = Adapt.adapt_structure(to, qv.mapping)
#     cell_coords = Adapt.adapt_structure(to, qv.cell_coords |> cu)
#     q_point = Adapt.adapt_structure(to, qv.q_point)
#     cv = Adapt.adapt_structure(to, qv.cv)
#     return StaticQuadratureView(mapping, cell_coords, q_point, cv)
# end

function Adapt.adapt_structure(to, grid::Grid)
    # map Int64 to Int32 to reduce number of registers
    cu_cells = grid.cells .|> (x -> Int32.(x.nodes)) .|> Quadrilateral |> cu
    cells = Adapt.adapt_structure(to, cu_cells)
    nodes = Adapt.adapt_structure(to, cu(grid.nodes))
    return GPUGrid(cells, nodes)
end


function Adapt.adapt_structure(to, iterator::CUDACellIterator)
    grid = Adapt.adapt_structure(to, iterator.grid)
    dh = Adapt.adapt_structure(to, iterator.dh)
    ncells = Adapt.adapt_structure(to, iterator.n_cells)
    return GPUCellIterator(dh, grid, ncells)
end


function _get_ndofs_cell(dh::DofHandler)
    ndofs_cell = [Int32(Ferrite.ndofs_per_cell(dh, i)) for i in 1:(dh |> Ferrite.get_grid |> Ferrite.getncells)]
    return ndofs_cell
end


_symbols_to_int32(symbols) = 1:length(symbols) .|> (sym -> convert(Int32, sym))

function Adapt.adapt_structure(to, sdh::SubDofHandler)
    @show "in sdh"
    cellset = Adapt.adapt_structure(to, sdh.cellset |> collect .|> (x -> convert(Int32, x)) |> cu)
    field_names = Adapt.adapt_structure(to, _symbols_to_int32(sdh.field_names) |> cu)
    field_interpolations = sdh.field_interpolations .|> (ip -> Adapt.adapt_structure(to, ip)) |> cu
    ndofs_per_cell = Adapt.adapt_structure(to, sdh.ndofs_per_cell)
    return GPUSubDofHandler(cellset, field_names, field_interpolations, ndofs_per_cell)
end

function Adapt.adapt_structure(to, dh::DofHandler)
    subdofhandlers = Adapt.adapt_structure(to,dh.subdofhandlers .|> (sdh -> Adapt.adapt_structure(to, sdh)) |> cu)
    cell_dofs = Adapt.adapt_structure(to, dh.cell_dofs .|> (x -> convert(Int32, x)) |> cu)
    cells = Adapt.adapt_structure(to, dh.grid.cells |> cu)
    offsets = Adapt.adapt_structure(to, dh.cell_dofs_offset .|> Int32 |> cu)
    nodes = Adapt.adapt_structure(to, dh.grid.nodes |> cu)
    #ndofs_cell = Adapt.adapt_structure(to, _get_ndofs_cell(dh) |> cu)
    cell_to_subdofhandler = Adapt.adapt_structure(to, dh.cell_to_subdofhandler .|> (x -> convert(Int32, x)) |> cu)
    return GPUDofHandler(subdofhandlers, cell_dofs, GPUGrid(cells, nodes), offsets, cell_to_subdofhandler)
end
