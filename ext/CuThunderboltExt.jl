module CuThunderboltExt

using Thunderbolt, CUDA

import CUDA:
    AbstractGPUVector, AbstractGPUArray

import Thunderbolt:
    UnPack.@unpack,
    SimpleMesh,
    SparseMatrixCSR, SparseMatrixCSC,
    AbstractSemidiscreteFunction, solution_size,
    AbstractPointwiseSolverCache

import Ferrite:
    AbstractDofHandler,
    AbstractGrid, AbstractCell

import Adapt: Adapt, adapt_structure, adapt

# ---------------------- Generic part ------------------------
function Adapt.adapt_structure(to, grid::SimpleMesh)
    return adapt_structure(to, grid.grid)
end

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

function Base.show(io::IO, mime, data::GPUDofHandlerData{sdim}) where sdim
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

function Adapt.adapt_structure(to::Type{<:CuArray}, dh::DofHandler{sdim}) where sdim
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

# Utility which holds partial information for assembly.
struct GPUGrid{sdim, C<:AbstractCell, T<:Real, CellDataType <: AbstractGPUVector{C}, NodeDataType <: AbstractGPUVector} <: AbstractGrid{sdim}
    cells::CellDataType
    nodes::NodeDataType
    #TODO subdomain info
end

function Base.show(io::IO, mime::MIME"text/plain", data::GPUGrid)
    _show(io, mime, data, 0)
end

function _show(io::IO, mime::MIME"text/plain", grid::GPUGrid{sdim, C, T}, indent) where {sdim, C, T}
    offset = " "^indent
    print(io, offset, "GPUGrid{sdim=", sdim, ", T=", T, "}")
    if isconcretetype(eltype(grid.cells))
        typestrs = [repr(eltype(grid.cells))]
    else
        typestrs = sort!(repr.(OrderedSet(typeof(x) for x in grid.cells)))
    end
    print(io, " with ")
    join(io, typestrs, '/')
    println(io, " cells and $(length(grid.nodes)) nodes")
end

function Adapt.adapt_structure(to::Type{<:CuArray}, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    node_type = typeof(first(grid.nodes))
    cells = Adapt.adapt_structure(CuVector{cell_type}, grid.cells)
    nodes = Adapt.adapt_structure(CuVector{node_type}, grid.nodes)
    #TODO subdomain info
    return GPUGrid{sdim, cell_type, T, typeof(cells), typeof(nodes)}(cells, nodes)
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

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where T = CUDA.zeros(T, solution_size(f))

function _gpu_pointwise_step_inner_kernel_wrapper!(f::F, t, Δt, cache::Thunderbolt.ForwardEulerCellSolverCache) where F <: PointwiseODEFunction # FIX LAST PARAMETER
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > Thunderbolt.solution_size(f) && return nothing
    Thunderbolt._pointwise_step_inner_kernel!(f, i, t, Δt, cache)
    # FIXME this crashes with unsupported dynamic function invocation (call to _pointwise_step_inner_kernel!)...
    # The code below is literally a copy-paste of the function contents

    # cell_model = f.ode
    # u_local    = @view cache.uₙmat[i, :]
    # du_local   = @view cache.dumat[i, :]
    # # TODO this should happen in rhs call below
    # @inbounds φₘ_cell = u_local[1]
    # @inbounds s_cell  = @view u_local[2:end]

    # # #TODO get spatial coordinate x and Cₘ
    # Thunderbolt.cell_rhs!(du_local, φₘ_cell, s_cell, nothing, t, cell_model)

    # for j in 1:length(u_local)
    #     u_local[j] += Δt*du_local[j]
    # end
    
    return nothing
end

Adapt.@adapt_structure Thunderbolt.ForwardEulerCellSolverCache
Adapt.@adapt_structure Thunderbolt.AdaptiveForwardEulerSubstepperCache

# This controls the outer loop over the ODEs
function Thunderbolt._pointwise_step_outer_kernel!(f::PointwiseODEFunction, t::Real, Δt::Real, cache::AbstractPointwiseSolverCache, ::CuVector)
    blocks = ceil(Int, f.npoints / cache.batch_size_hint)
    @cuda threads=cache.batch_size_hint blocks _gpu_pointwise_step_inner_kernel_wrapper!(f, t, Δt, cache) # || return false
    return true
end

end
