module CuThunderboltExt

using Thunderbolt

import CUDA:
    CUDA, CuArray, CuVector, CUSPARSE,
    threadIdx, blockIdx, blockDim, @cuda,
    CUDABackend, launch_configuration

import Thunderbolt:
    UnPack.@unpack,
    SimpleMesh,
    SparseMatrixCSR, SparseMatrixCSC,
    AbstractSemidiscreteFunction, AbstractPointwiseFunction, solution_size,
    AbstractPointwiseSolverCache,
    GPUDofHandlerData, GPUSubDofHandlerData, GPUDofHandler,
    GPUGrid

import Ferrite:
    AbstractDofHandler

import Adapt:
    Adapt, adapt_structure, adapt

# ---------------------- Generic part ------------------------
function _convert_subdofhandler_to_gpu(cell_dofs, cell_dof_soffset, sdh::SubDofHandler)
    GPUSubDofHandler(
        cell_dofs,
        cell_dofs_offset,
        adapt(typeof(cell_dofs), collect(sdh.cellset)),
        Tuple(sym for sym in sdh.field_names),
        Tuple(sym for sym in sdh.field_n_components),
        sdh.ndofs_per_cell.x,
    )
end

function Adapt.adapt_structure(to::Type{CUDABackend}, dh::DofHandler{sdim}) where sdim
    grid             = adapt_structure(to, dh.grid)
    # field_names      = Tuple(sym for sym in dh.field_names)
    IndexType        = eltype(dh.cell_dofs)
    IndexVectorType  = CuVector{IndexType}
    cell_dofs        = adapt(IndexVectorType, dh.cell_dofs)
    cell_dofs_offset = adapt(IndexVectorType, dh.cell_dofs_offset)
    cell_to_sdh      = adapt(IndexVectorType, dh.cell_to_subdofhandler)
    subdofhandlers   = Tuple(i->_convert_subdofhandler_to_gpu(cell_dofs, cell_dofs_offset, sdh) for sdh in dh.subdofhandlers)
    gpudata = GPUDofHandlerData(
        grid,
        subdofhandlers,
        # field_names,
        cell_dofs,
        cell_dofs_offset,
        cell_to_sdh,
        dh.ndofs.x,
    )
    return GPUDofHandler(dh, gpudata)
end



function Adapt.adapt_structure(to::Type{CUDABackend}, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    node_type = typeof(first(grid.nodes))
    cells = Adapt.adapt_structure(to, grid.cells)
    nodes = Adapt.adapt_structure(to, grid.nodes)
    #TODO subdomain info
    return GPUGrid{sdim, cell_type, T, typeof(cells), typeof(nodes)}(cells, nodes)
end

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
function Thunderbolt._pointwise_step_outer_kernel!(f::AbstractPointwiseFunction, t::Real, Δt::Real, cache::AbstractPointwiseSolverCache, ::CuVector)
    kernel = @cuda launch=false _gpu_pointwise_step_inner_kernel_wrapper!(f.ode, t, Δt, cache) # || return false
    config = launch_configuration(kernel.fun)
    threads = min(f.npoints, config.threads)
    blocks =  cld(f.npoints, threads)
    kernel(f.ode, t, Δt, cache;  threads, blocks)
    return true
end

_create_sparsity_pattern(dh::GPUDofHandler, A::SparseMatrixCSR, ::CuVector) = CuSparseMatrixCSR(A)
_create_sparsity_pattern(dh::GPUDofHandler, A::SparseMatrixCSC, ::CuVector) = CuSparseMatrixCSC(A)

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where T = CUDA.zeros(T, solution_size(f))
Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, dh::DofHandler) where T                  = CUDA.zeros(T, ndofs(dh))

function Thunderbolt.create_system_matrix(SpMatType::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}}, dh::AbstractDofHandler)
    # FIXME in general the pattern is not symmetric
    Acpu      = create_sparsity_pattern(dh)
    colptrgpu = CuArray(Acpu.colptr)
    rowvalgpu = CuArray(Acpu.rowval)
    nzvalgpu  = CuArray(Acpu.nzval)
    return SpMatType(colptrgpu, rowvalgpu, nzvalgpu, (Acpu.m, Acpu.n))
end

Thunderbolt.__add_to_vector!(b::Vector, a::CuVector) = b .+= Vector(a)
Thunderbolt.__add_to_vector!(b::CuVector, a::Vector) = b .+= CuVector(a)

end
