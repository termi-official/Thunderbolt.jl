
## shared memory allocation ##
struct DynamicSharedMemFunction{N, Tv <: Real, Ti <: Integer}
    mem_size::NTuple{N, Ti}
    offset::Ti
end


function (dsf::DynamicSharedMemFunction{N, Tv, Ti})() where {N, Tv, Ti}
    mem_size = dsf.mem_size
    offset = dsf.offset
    return CUDA.@cuDynamicSharedMem(Tv, mem_size, offset)
end

struct SharedMemAlloc{N, M, Tv <: Real, Ti <: Integer} <: AbstractDeviceSharedMemAlloc
    Ke::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    fe::DynamicSharedMemFunction{M, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

Thunderbolt.FerriteUtils.mem_size(alloc::SharedMemAlloc) = alloc.tot_mem_size


function _can_use_dynshmem(required_shmem::Integer)
    dev = device()
    MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return required_shmem < MAX_DYN_SHMEM
end


function Thunderbolt.FerriteUtils.try_allocate_shared_mem(::Type{FullObject{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * (n_basefuncs) * n_basefuncs + sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    Ke = DynamicSharedMemFunction{3, Tv, Ti}((block_dim, n_basefuncs, n_basefuncs), convert(Ti, 0))
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, sizeof(Tv) * block_dim * n_basefuncs * n_basefuncs))
    return SharedMemAlloc(Ke, fe, shared_mem)
end

struct SharedRHSMemAlloc{N, Tv <: Real, Ti <: Integer} <: AbstractDeviceSharedMemAlloc
    fe::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

function Thunderbolt.FerriteUtils.try_allocate_shared_mem(::Type{RHSObject{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, 0))
    return SharedRHSMemAlloc( fe, shared_mem)
end

function Thunderbolt.FerriteUtils.try_allocate_shared_mem(::Type{JacobianObject{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    error("Not implemented")
end


## global memory allocation ##
struct GlobalMemAlloc{LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractDeviceGlobalMemAlloc
    Kes::LOCAL_MATRICES ## global level allocation (i.e. memory for all blocks -> 3rd order tensor)
    fes::LOCAL_VECTORS  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

Thunderbolt.FerriteUtils.cellke(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.Kes[e, :, :]
Thunderbolt.FerriteUtils.cellfe(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.fes[e, :]


function Thunderbolt.FerriteUtils.allocate_global_mem(::Type{FullObject{Tv}}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    Kes = CUDA.zeros(Tv, n_cells, n_basefuncs, n_basefuncs)
    fes = CUDA.zeros(Tv, n_cells, n_basefuncs)
    return GlobalMemAlloc(Kes, fes)
end


struct GlobalRHSMemAlloc{LOCAL_VECTORS} <: AbstractDeviceGlobalMemAlloc
    fes::LOCAL_VECTORS  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

Thunderbolt.FerriteUtils.cellfe(alloc::GlobalRHSMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.fes[e, :]


function Thunderbolt.FerriteUtils.allocate_global_mem(::Type{RHSObject{Tv}}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    fes = CUDA.zeros(Tv, n_cells, n_basefuncs)
    return GlobalRHSMemAlloc( fes)
end