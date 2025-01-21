abstract type AbstractMemAllocObjects{Tv <: Real} end
struct RHSObject{Tv<:Real} <:AbstractMemAllocObjects{Tv}  end
struct JacobianObject{Tv<:Real} <:AbstractMemAllocObjects{Tv}  end
struct FullObject{Tv<:Real} <:AbstractMemAllocObjects{Tv}  end

abstract type AbstractMemAlloc end
abstract type AbstractDeviceMemAlloc <: AbstractMemAlloc end
abstract type AbstractDeviceSharedMemAlloc <: AbstractDeviceMemAlloc end
abstract type AbstractDeviceGlobalMemAlloc <: AbstractDeviceMemAlloc end

# interfaces
mem_size(alloc::AbstractDeviceSharedMemAlloc) = error("please provide concrete implementation for $(typeof(alloc)).")

function try_allocate_shared_mem(::Type{AbstractMemAllocObjects{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    error("please provide concrete implementation for $(typeof(AbstractMemAllocObjects{Tv})).")
end

cellke(alloc::AbstractDeviceGlobalMemAlloc, ::Ti) where {Ti <: Integer} = error("please provide concrete implementation for $(typeof(alloc)).")
cellfe(alloc::AbstractDeviceGlobalMemAlloc, ::Ti) where {Ti <: Integer} = error("please provide concrete implementation for $(typeof(alloc)).")

function allocate_global_mem(::Type{AbstractMemAllocObjects{Tv}}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
   error("please provide concrete implementation for $(typeof(AbstractMemAllocObjects{Tv})).")
end