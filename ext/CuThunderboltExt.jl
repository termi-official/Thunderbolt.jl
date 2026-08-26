module CuThunderboltExt

# CUDA support is limited to what this extension declares:
#   * the pointwise cell-model solve, whose outer loop becomes a CUDA kernel launch;
#   * `CuVector`/`CuSparseMatrix` system allocation for the solver interface;
#   * moving a coefficient's coordinate vector to the device.
# Assembly on the GPU goes through `FerriteOperators`' device seam, not through here.

using Thunderbolt

import CUDA:
    CUDA,
    CuArray,
    CuVector,
    CUSPARSE,
    blockDim,
    blockIdx,
    threadIdx,
    @cuda,
    launch_configuration

import Thunderbolt:
    AbstractSemidiscreteFunction,
    AbstractPointwiseFunction,
    AbstractPointwiseSolverCache,
    solution_size

import Ferrite: AbstractDofHandler

########################
## Pointwise solvers  ##
########################

function _gpu_pointwise_step_inner_kernel_wrapper!(f, t, Δt, cache::AbstractPointwiseSolverCache)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > size(cache.dumat, 1) && return nothing
    Thunderbolt._pointwise_step_inner_kernel!(f, i, t, Δt, cache)
    return nothing
end

# This controls the outer loop over the ODEs
function Thunderbolt._pointwise_step_outer_kernel!(
    f::AbstractPointwiseFunction,
    t::Real,
    Δt::Real,
    cache::AbstractPointwiseSolverCache,
    ::Union{<:CuVector, SubArray{<:Any, 1, <:CuVector}},
)
    kernel = @cuda launch=false _gpu_pointwise_step_inner_kernel_wrapper!(f.ode, t, Δt, cache)
    config = launch_configuration(kernel.fun)
    threads = min(f.npoints, config.threads)
    blocks = cld(f.npoints, threads)
    kernel(f.ode, t, Δt, cache; threads, blocks)
    return true
end

########################
## System allocation  ##
########################

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where {T} = CUDA.zeros(T, solution_size(f))
Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, dh::DofHandler) where {T}                  = CUDA.zeros(T, ndofs(dh))

function Thunderbolt.create_system_matrix(
    SpMatType::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}},
    dh::AbstractDofHandler,
)
    # FIXME in general the pattern is not symmetric
    Acpu      = allocate_matrix(dh)
    colptrgpu = CuArray(Acpu.colptr)
    rowvalgpu = CuArray(Acpu.rowval)
    nzvalgpu  = CuArray(Acpu.nzval)
    return SpMatType(colptrgpu, rowvalgpu, nzvalgpu, (Acpu.m, Acpu.n))
end

Thunderbolt.adapt_vector_type(::Type{<:CuVector}, v::VT) where {VT <: Vector} = CuVector(v)

end
