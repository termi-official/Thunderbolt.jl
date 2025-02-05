
# # Utility which holds partial information for assembly.
struct DeviceSubDofHandlerData{VEC_IP,IndexType, IndexVectorType <: AbstractVector{IndexType},Ti<:Integer} <: Ferrite.AbstractDofHandler
    cellset::IndexVectorType
    field_names::IndexVectorType
    field_interpolations::VEC_IP 
    ndofs_per_cell::Ti
end


# Utility which holds partial information for assembly.
struct DeviceDofHandlerData{sdim, G<:Ferrite.AbstractGrid{sdim}, #=nfields,=# SDHTupleType, IndexType, IndexVectorType <: AbstractVector{IndexType},Ti<: Integer} <: Ferrite.AbstractDofHandler
    grid::G
    subdofhandlers::SDHTupleType
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    cell_to_subdofhandler::IndexVectorType
    ndofs::Ti
end

struct DeviceDofHandler{ GPUDataType} <: Ferrite.AbstractDofHandler
    gpudata::GPUDataType
end

Ferrite.isclosed(::DeviceDofHandler) = true

Ferrite.allocate_matrix(dh::DeviceDofHandler) = _allocate_matrix(dh, allocate_matrix(dh.dh), dh.gpudata.cellset)

function Ferrite.ndofs_per_cell(dh::DeviceDofHandlerData, cell::Ti) where {Ti <: Integer}
    sdhidx = dh.cell_to_subdofhandler[cell]
    sdhidx ∉ 1:length(dh.subdofhandlers) && return 0 # Dof handler is just defined on a subdomain
    return ndofs_per_cell(dh.subdofhandlers[sdhidx])
end
Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandlerData) = sdh.ndofs_per_cell
cell_dof_offset(dh::DeviceDofHandlerData, i::Ti) where {Ti<:Integer} = dh.cell_dofs_offset[i]
Ferrite.get_grid(dh::DeviceDofHandlerData) = dh.grid

function Ferrite.celldofs(dh::DeviceDofHandlerData, i::Ti) where {Ti<:Integer}
    offset = cell_dof_offset(dh, i)
    ndofs = ndofs_per_cell(dh, i)
    view = @view dh.cell_dofs[offset:(offset + ndofs - convert(Ti, 1))]
    return view
end
