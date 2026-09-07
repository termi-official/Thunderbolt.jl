module CuThunderboltExt

# CUDA support is limited to what this extension declares:
#   * the pointwise cell-model solve, whose outer loop becomes a CUDA kernel launch;
#   * `CuVector`/`CuSparseMatrix` system allocation for the solver interface;
#   * mirroring a host-assembled bilinear operator into a device matrix;
#   * moving a coefficient's coordinate vector to the device.
# Assembly on the GPU goes through `FerriteOperators`' device seam, not through here.

using Thunderbolt

import CUDA:
    CUDA, CuArray, CuVector, CUSPARSE, blockDim, blockIdx, threadIdx, @cuda, launch_configuration

import Thunderbolt:
    AbstractSemidiscreteFunction,
    AbstractPointwiseFunction,
    AbstractPointwiseSolverCache,
    AbstractBilinearIntegrator,
    AbstractCPUDevice,
    AssemblyStrategy,
    FullAssembly,
    SequentialScheduling,
    MirroredBilinearOperator,
    num_states,
    solution_size

import FerriteOperators

import Ferrite: AbstractDofHandler

import SparseArrays: SparseMatrixCSC

##########################
## Pointwise solvers
##########################

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
    npoints = length(f.associated_states) ÷ num_states(f.ode)
    kernel = @cuda launch=false _gpu_pointwise_step_inner_kernel_wrapper!(f.ode, t, Δt, cache)
    config = launch_configuration(kernel.fun)
    threads = min(npoints, config.threads)
    blocks = cld(npoints, threads)
    kernel(f.ode, t, Δt, cache; threads, blocks)
    return true
end

##########################
## System allocation
##########################

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where {T} = CUDA.zeros(T, solution_size(f))
Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, dh::DofHandler) where {T}                  = CUDA.zeros(T, ndofs(dh))

function Thunderbolt.create_system_matrix(
    ::Type{<:CUSPARSE.CuSparseMatrixCSC{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsc = convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh))
    return CUSPARSE.CuSparseMatrixCSC{Tv, Ti}(
        CuVector{Ti}(Acsc.colptr),
        CuVector{Ti}(Acsc.rowval),
        CuVector{Tv}(Acsc.nzval),
        size(Acsc),
    )
end

# The CSC arrays are handed to the CSR type unchanged, so what this returns is the CSR of `Aᵀ`, and
# every operator assembled into it is assumed symmetric -- the same assumption
# `create_system_matrix(::Type{<:ThreadedSparseMatrixCSR}, dh)` in `src/solver/interface.jl` carries,
# for the same reason: what the affine backward Euler stage combines is the entrywise `nonzeros`
# correspondence between this matrix and the operators mirrored into the same pattern, and CUSPARSE's
# faithful CSC→CSR conversion would reorder the entries and break it. The two have to be changed
# together.
function Thunderbolt.create_system_matrix(
    ::Type{<:CUSPARSE.CuSparseMatrixCSR{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsc = convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh))
    return CUSPARSE.CuSparseMatrixCSR{Tv, Ti}(
        CuVector{Ti}(Acsc.colptr),
        CuVector{Ti}(Acsc.rowval),
        CuVector{Tv}(Acsc.nzval),
        reverse(size(Acsc)),
    )
end

##########################
## Host assembly, device A
##########################

function Thunderbolt.setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}},
    dh::AbstractDofHandler,
)
    return MirroredBilinearOperator(
        Thunderbolt.setup_operator(strategy, integrator, dh),
        Thunderbolt.create_system_matrix(system_matrix_type, dh),
    )
end

##########################
## Cross device vectors
##########################

# A source term assembled on one device added into a solution vector living on the other.
FerriteOperators.__add_to_vector!(b::Vector, a::CuVector) = b .+= Vector(a)
FerriteOperators.__add_to_vector!(b::CuVector, a::Vector) = b .+= CuVector(a)

Thunderbolt.adapt_vector_type(::Type{<:CuVector}, v::VT) where {VT <: Vector} = CuVector(v)

end
