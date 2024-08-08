# Utility which holds partial information for assembly.
struct GPUSubDofHandlerData{IndexType, IndexVectorType <: AbstractGPUVector{IndexType}} <: AbstractDofHandler
    # Relevant fields from GPUDofHandler
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    # Flattened cellset
    cellset::IndexVectorType
    # Dunno if we need this
    ndofs_per_cell::Int
end

# Utility which holds partial information for assembly.
struct GPUDofHandlerData{sdim, G<:AbstractGrid{sdim}, #=nfields,=# SDHTupleType, IndexType, IndexVectorType <: AbstractGPUVector{IndexType}} <: AbstractDofHandler
    grid::G
    subdofhandlers::SDHTupleType
    # field_names::SVector{Symbol, nfields}
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    cell_to_subdofhandler::IndexVectorType
    # grid::G # We do not need this explicitly on the GPU
    ndofs::Int
end

function Base.show(io::IO, mime::MIME"text/plain", data::GPUDofHandlerData{sdim}) where sdim
    _show(io, mime, data, 0)
end

function _show(io::IO, mime::MIME"text/plain", data::GPUDofHandlerData{sdim}, indent::Int) where sdim
    offset = " "^indent
    println(io, offset, "GPUDofHandlerData{sdim=", sdim, "}")
    _show(io, mime, data.grid, indent+2)
    println(io, offset, "  SubDofHandlers: ", length(data.subdofhandlers))
end

struct GPUDofHandler{DHType <: AbstractDofHandler, GPUDataType} <: AbstractDofHandler
    dh::DHType
    gpudata::GPUDataType
end

function Base.show(io::IO, mime::MIME"text/plain", dh_::GPUDofHandler)
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

Ferrite.isclosed(::GPUDofHandler) = true

# TODO needs an update after https://github.com/Ferrite-FEM/Ferrite.jl/pull/888 is merged
Ferrite.allocate_matrix(dh::GPUDofHandler) = _allocate_matrix(dh, allocate_matrix(dh.dh), dh.gpudata.cellset)
