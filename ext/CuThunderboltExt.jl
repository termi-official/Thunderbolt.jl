module CuThunderboltExt

using Thunderbolt, CUDA

import Thunderbolt: SimpleMesh
import Ferrite: AbstractDofHandler, AbstractGrid, AbstractCell

import Adapt: Adapt, adapt_structure, adapt

# ---------------------- Generic part ------------------------
function Adapt.adapt_structure(to, grid::SimpleMesh)
    return adapt_structure(to, grid.grid)
end

# Utility which holds partial information for assembly.
struct GPUSubDofHandlerData{IndexType, IndexVectorType <: AbstractGPUArray{IndexType,1}} <: AbstractDofHandler
    # Relevant fields from GPUDofHandler
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    # Flattened cellset
    cellset::IndexVectorType
    # Dunno if we need this
    ndofs_per_cell::Int
end

# Utility which holds partial information for assembly.
struct GPUDofHandlerData{sdim, G<:AbstractGrid{sdim}, #=nfields,=# SDHTupleType, IndexType, IndexVectorType <: AbstractGPUArray{IndexType,1}} <: AbstractDofHandler
    grid::G
    subdofhandlers::SDHTupleType
    # field_names::SVector{Symbol, nfields}
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    cell_to_subdofhandler::IndexVectorType
    # grid::G # We do not need this explicitly on the GPU
    ndofs::Int
end

struct GPUDofHandler{sdim, DHType <: AbstractDofHandler{sdim}, GPUDataType} <: AbstractDofHandler{sdim}
    dh::DHType
    gpudata::GPUDataType
end

Ferrite.isclosed(::GPUDofHandler) = true

function _convert_subdofhandler_to_gpu(to, cell_dofs, cell_dof_soffset, sdh::SubDofHandler)
    GPUSubDofHandler(
        cell_dofs,
        cell_dofs_offset,
        adapt(to, collect(sdh.cellset)),
        Tuple(sym for sym in sdh.field_names),
        Tuple(sym for sym in sdh.field_n_components),
        sdh.ndofs_per_cell.x,
    )
end

function Adapt.adapt_structure(to, dh::DofHandler{sdim}) where sdim
    grid = adapt_structure(to, dh.grid)
    field_names      = Tuple(sym for sym in dh.field_names)
    cell_dofs        = adapt(to, dh.cell_dofs)
    cell_dofs_offset = adapt(to, dh.cell_dofs_offsett)
    cellset          = adapt(to, collect(dh.cellset))
    subdofhandlers   = Tuple(i->_convert_subdofhandler_to_gpu(to, cell_dofs, cell_dofs_offset, sdh) for sdh in dh.subdofhandlers)
    data = GPUDofHandlerData(
        subdofhandlers,
        field_names,
        cell_dofs,
        cell_dofs_offset,
        cell_to_subdofhandler,
        dh.ndofs.x,
    )
    return GPUDofHandler{sdim, typeof(subdofhandlers), eltype(cell_dofs), typeof(cell_dofs)}(
        dh,
        data
    )
end

# Utility which holds partial information for assembly.
struct GPUGrid{sdim, C<:AbstractCell, T<:Real, CellDataType <: AbstractGPUArray{C, 1}, NodeDataType <: AbstractGPUArray{C, 1}} <: AbstractGrid{sdim}
    cells::CellDataType
    nodes::NodeDataType
    #TODO subdomain info
end

function Adapt.adapt_structure(to, grid::Grid)
    cells = adapt(to, grid.cells)
    nodes = adapt(to, grid.nodes)
    #TODO subdomain info
    return GPUGrid(cells, nodes)
end

function Thunderbolt.setup_operator(protocol::Thunderbolt.AnalyticalTransmembraneStimulationProtocol, solver::Thunderbolt.AbstractSolver, dh::GPUDofHandler, field_name::Symbol, qr)
    ip = dh.dh.subdofhandlers[1].field_interpolations[1]
    ip_g = Ferrite.geometric_interpolation(typeof(getcells(Ferrite.get_grid(dh), 1)))
    qr = QuadratureRule{Ferrite.getrefshape(ip_g)}(Ferrite.getorder(ip_g)+1)
    cv = CellValues(qr, ip, ip_g) # TODO replace with GPUCellValues
    return PEALinearOperator(
        zeros(ndofs(dh)),
        AnalyticalCoefficientElementCache(
            protocol.f,
            protocol.nonzero_intervals,
            cv,
        ),
        dh,
    )
end

# TODO needs an update after https://github.com/Ferrite-FEM/Ferrite.jl/pull/888 is merged
Ferrite.create_sparsity_pattern(dh::GPUDofHandler) = _create_sparsity_pattern(dh, create_sparsity_pattern(dh.dh), dh.gpudata.cellset)

_create_sparsity_pattern(dh::GPUDofHandler, A::SparseMatrixCSR, ::CuVector) = CuSparseMatrixCSR(A)
_create_sparsity_pattern(dh::GPUDofHandler, A::SparseMatrixCSC, ::CuVector) = CuSparseMatrixCSC(A)

end
