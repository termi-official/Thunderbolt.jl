
# Utility which holds partial information for assembly.
struct DeviceSubDofHandlerData{VEC_IP,IndexType, IndexVectorType <: AbstractVector{IndexType},Ti<:Integer} <: Ferrite.AbstractDofHandler
    # Relevant fields from GPUDofHandler
    #cell_dofs::IndexVectorType # why we need this?
    #cell_dofs_offset::IndexVectorType # why we need this?
    # Flattened cellset
    cellset::IndexVectorType
    field_names::IndexVectorType
    field_interpolations::VEC_IP
    ndofs_per_cell::Ti
end

# Utility which holds partial information for assembly.
struct DeviceDofHandlerData{sdim, G<:Ferrite.AbstractGrid{sdim}, #=nfields,=# SDHTupleType, IndexType, IndexVectorType <: AbstractVector{IndexType},Ti<: Integer} <: Ferrite.AbstractDofHandler
    grid::G
    subdofhandlers::SDHTupleType
    # field_names::SVector{Symbol, nfields}
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    cell_to_subdofhandler::IndexVectorType
    # grid::G # We do not need this explicitly on the GPU
    ndofs::Ti
end

function Base.show(io::IO, mime::MIME"text/plain", data::DeviceDofHandlerData{sdim}) where sdim
    _show(io, mime, data, 0)
end

function _show(io::IO, mime::MIME"text/plain", data::DeviceDofHandlerData{sdim}, indent::Int) where sdim
    offset = " "^indent
    println(io, offset, "DeviceDofHandlerData{sdim=", sdim, "}")
    _show(io, mime, data.grid, indent+2)
    println(io, offset, "  SubDofHandlers: ", length(data.subdofhandlers))
end

# struct GPUDofHandler{DHType <: Ferrite.AbstractDofHandler, GPUDataType} <: Ferrite.AbstractDofHandler
#     #dh::DHType #Why do we need this? already all info is in gpudata
#     gpudata::GPUDataType
# end
struct DeviceDofHandler{ GPUDataType} <: Ferrite.AbstractDofHandler
    #dh::DHType #Why do we need this? already all info is in gpudata
    gpudata::GPUDataType
end

function Base.show(io::IO, mime::MIME"text/plain", dh_::DeviceDofHandler)
    dh = dh_.dh
    println(io, "GPUDofHandler")
    if length(dh.subdofhandlers) == 1
        Ferrite._print_field_information(io, mime, dh.subdofhandlers[1])
    else
        println(io, "  Fields:")
        for fieldname in getfieldnames(dh)
            ip = getfieldinterpolation(dh, find_field(dh, fieldname))
            if ip isa ScalarInterpolation
                field_type = "scalar"
            elseif ip isa VectorInterpolation
                _getvdim(::VectorInterpolation{vdim}) where vdim = vdim
                field_type = "Vec{$(_getvdim(ip))}"
            end
            println(io, "    ", repr(fieldname), ", ", field_type)
        end
    end
    if !Ferrite.isclosed(dh)
        println(io, "  Not closed!")
    else
        println(io, "  Total dofs: ", ndofs(dh))
    end
    _show(io, mime, dh_.gpudata, 2)
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
