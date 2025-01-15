module CuThunderboltExt

using Thunderbolt

import CUDA:
    CUDA, CuArray, CuVector, CUSPARSE,
    threadIdx, blockIdx, blockDim, @cuda, @cushow,
    CUDABackend, launch_configuration, device, cu

import Thunderbolt:
    UnPack.@unpack,
    SimpleMesh,
    SparseMatrixCSR, SparseMatrixCSC,
    AbstractSemidiscreteFunction, AbstractPointwiseFunction, solution_size,
    AbstractPointwiseSolverCache,assemble_element!,
    LinearOperator,AbstractOperatorKernel,QuadratureRuleCollection,
    AnalyticalCoefficientElementCache,AnalyticalCoefficientCache,CartesianCoordinateSystemCache,
    setup_element_cache,update_operator!,init_linear_operator

import Thunderbolt.FerriteUtils:
    AbstractGlobalMemAlloc, AbstractSharedMemAlloc,
    StaticInterpolationValues,StaticCellValues, try_allocate_shared_mem,
    CellIterator,allocate_global_mem, RHSObject, mem_size,
    GPUDofHandlerData, GPUSubDofHandlerData, GPUDofHandler, GPUGrid,
    cellfe,celldofs

import Ferrite:
    AbstractDofHandler,get_grid,CellIterator

import Adapt:
    Adapt, adapt_structure, adapt

# ---------------------- Generic part ------------------------


# function Thunderbolt.setup_operator(protocol::Thunderbolt.AnalyticalTransmembraneStimulationProtocol, solver::Thunderbolt.AbstractSolver, dh::GPUDofHandler, field_name::Symbol, qr)
#     ip = dh.dh.subdofhandlers[1].field_interpolations[1]
#     ip_g = Ferrite.geometric_interpolation(typeof(getcells(Ferrite.get_grid(dh), 1)))
#     qr = QuadratureRule{Ferrite.getrefshape(ip_g)}(Ferrite.getorder(ip_g)+1)
#     cv = CellValues(qr, ip, ip_g) # TODO replace with GPUCellValues
#     return PEALinearOperator(
#         zeros(ndofs(dh)),
#         AnalyticalCoefficientElementCache(
#             protocol.f,
#             protocol.nonzero_intervals,
#             cv,
#         ),
#         dh,
#     )
# end

# Pointwise cuda solver wrapper
function _gpu_pointwise_step_inner_kernel_wrapper!(f, t, Δt, cache::AbstractPointwiseSolverCache)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > size(cache.dumat, 1) && return nothing
    Thunderbolt._pointwise_step_inner_kernel!(f, i, t, Δt, cache)
    return nothing
end

# This controls the outer loop over the ODEs
function Thunderbolt._pointwise_step_outer_kernel!(f::AbstractPointwiseFunction, t::Real, Δt::Real, cache::AbstractPointwiseSolverCache, ::Union{<:CuVector, SubArray{<:Any,1,<:CuVector}})
    kernel = @cuda launch=false _gpu_pointwise_step_inner_kernel_wrapper!(f.ode, t, Δt, cache) # || return false
    config = launch_configuration(kernel.fun)
    threads = min(f.npoints, config.threads)
    blocks =  cld(f.npoints, threads)
    kernel(f.ode, t, Δt, cache;  threads, blocks)
    return true
end

_allocate_matrix(dh::GPUDofHandler, A::SparseMatrixCSR, ::CuVector) = CuSparseMatrixCSR(A)
_allocate_matrix(dh::GPUDofHandler, A::SparseMatrixCSC, ::CuVector) = CuSparseMatrixCSC(A)

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where T = CUDA.zeros(T, solution_size(f))
Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, dh::DofHandler) where T                  = CUDA.zeros(T, ndofs(dh))

function Thunderbolt.create_system_matrix(SpMatType::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}}, dh::AbstractDofHandler)
    # FIXME in general the pattern is not symmetric
    Acpu      = allocate_matrix(dh)
    colptrgpu = CuArray(Acpu.colptr)
    rowvalgpu = CuArray(Acpu.rowval)
    nzvalgpu  = CuArray(Acpu.nzval)
    return SpMatType(colptrgpu, rowvalgpu, nzvalgpu, (Acpu.m, Acpu.n))
end

Thunderbolt.__add_to_vector!(b::Vector, a::CuVector) = b .+= Vector(a)
Thunderbolt.__add_to_vector!(b::CuVector, a::Vector) = b .+= CuVector(a)

function Thunderbolt.adapt_vector_type(::Type{<:CuVector}, v::VT) where {VT <: Vector}
    return CuVector(v)
end

include("cuda/cuda_operator.jl")
include("cuda/cuda_adapt.jl")

end
