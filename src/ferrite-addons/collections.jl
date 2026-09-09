function FerriteOperators.getquadraturerule(
    qrc::FerriteOperators.QuadratureRuleCollection,
    cell::InterfaceCell,
)
    return getquadraturerule(qrc, cell.here)
end

"""
    element_precision(qr) -> Type

The scalar type an element built on `qr` evaluates in — the precision the
integrator elected through its quadrature collection, read off the rule that
collection produced. `CellValues(qr, ip, ip_geo)` is `Float64` whatever the rule
carries, so every cache built here spells `CellValues(element_precision(qr), qr,
ip, ip_geo)`.

Read off the RULE rather than the collection so `src/` keeps loading against
FerriteOperators versions whose collections are `Float64` only.
"""
# TODO(FO floor >= 0.5): once the FerriteOperators floor carries `element_value_type`
# unconditionally, collapse this whole precision-plumbing seam:
#   (1) delete `element_precision`/`element_matrix_buffer`/`element_vector_buffer` (collections.jl:8-39);
#   (2) delete the nine `allocate_element_matrix`/`allocate_element_unknown_vector`/
#       `allocate_element_residual_vector` overrides (mass.jl:85-90, diffusion.jl:87-92,
#       analytical_coefficient.jl:67-72) — FO's own defaults already read `element_value_type`;
#   (3) add three `FerriteOperators.element_value_type` methods instead (`BilinearMassElementCache`
#       and `BilinearDiffusionElementCache` via `.cellvalues`, `AnalyticalCoefficientElementCache`
#       via `.cv`);
#   (4) `element_precision` → `element_value_type` at mass.jl:100, diffusion.jl:100,
#       electrophysiology.jl:281;
#   (5) bump the FerriteOperators compat bound in Project.toml.
# FO's `allocate_element_*` defaults also pad what they return for a `global_dofs` declaration; any
# override kept past this collapse has to carry that same padding responsibility itself.
element_precision(qr::QuadratureRule) = eltype(Ferrite.getweights(qr))
element_precision(cv::Ferrite.AbstractValues) = eltype(Ferrite.shape_value_type(cv))

"""
    element_matrix_buffer(cv, sdh)
    element_vector_buffer(cv, sdh)

The element-local buffers of a cache evaluating through `cv`, in that values
object's own precision — `Float32` values must not accumulate into `Float64`
buffers.

Spelled on FerriteOperators' `allocate_element_*` hooks rather than on its
element precision trait, which `src/` cannot name while it also loads against
FerriteOperators 0.4; against that version these return exactly the default they
replace.
"""
element_matrix_buffer(cv, sdh) =
    zeros(element_precision(cv), ndofs_per_cell(sdh), ndofs_per_cell(sdh))
@doc (@doc element_matrix_buffer) element_vector_buffer(cv, sdh) =
    zeros(element_precision(cv), ndofs_per_cell(sdh))

"""
    InterpolationCollection

A collection of compatible interpolations over some (possilby different) cells.
"""
abstract type InterpolationCollection end

"""
    ScalarInterpolationCollection

A collection of compatible scalar-valued interpolations over some (possilby different) cells.
"""
abstract type ScalarInterpolationCollection <: InterpolationCollection end

"""
    VectorInterpolationCollection

A collection of compatible vector-valued interpolations over some (possilby different) cells.
"""
abstract type VectorInterpolationCollection <: InterpolationCollection end

struct InterfaceCollection{IPC} <: InterpolationCollection
    ipc::IPC
end

getorder(ic::InterfaceCollection) = getorder(ic.ipc)

function getinterpolation(ic::InterfaceCollection, cell::InterfaceCell)
    return InterfaceCellInterpolation(getinterpolation(ic.ipc, cell.here))
end

# Wildcard
"""
    getinterpolation(ipc::InterpolationCollection, cell::AbstractCell)
    getinterpolation(ipc::InterpolationCollection, ::Type{<:AbstractRefShape})
    getinterpolation(ipc::InterpolationCollection, sdh::SubDofHandler)

The collection's interpolation for a reference shape: the cell's own, or -- the form the
discretization uses -- the shape of the subdomain's first cell, shared by every cell of a
`SubDofHandler`.
"""
getinterpolation(ipc::InterpolationCollection, sdh::SubDofHandler) =
    getinterpolation(ipc, get_first_cell(sdh))

"""
    LagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct LagrangeCollection{order} <: ScalarInterpolationCollection end

getorder(::LagrangeCollection{order}) where {order} = order
getinterpolation(
    lc::LagrangeCollection{order},
    cell::AbstractCell{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()
getinterpolation(
    lc::LagrangeCollection{order},
    ::Type{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()

"""
    DiscontinuousLagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct DiscontinuousLagrangeCollection{order} <: ScalarInterpolationCollection end

getorder(::DiscontinuousLagrangeCollection{order}) where {order} = order
getinterpolation(
    lc::DiscontinuousLagrangeCollection{order},
    cell::AbstractCell{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()
getinterpolation(
    lc::DiscontinuousLagrangeCollection{order},
    ::Type{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()


"""
    VectorizedInterpolationCollection{order} <: InterpolationCollection

A collection of fixed-order vectorized Lagrange interpolations across different cell types.
"""
struct VectorizedInterpolationCollection{vdim, IPC <: ScalarInterpolationCollection} <:
       VectorInterpolationCollection
    base::IPC
    function VectorizedInterpolationCollection{vdim}(
        ip::SIPC,
    ) where {vdim, SIPC <: ScalarInterpolationCollection}
        return new{vdim, SIPC}(ip)
    end
end

Base.:(^)(ip::ScalarInterpolationCollection, vdim::Int) =
    VectorizedInterpolationCollection{vdim}(ip)

getorder(ipc::VectorizedInterpolationCollection) = getorder(ipc.base)
getinterpolation(
    ipc::VectorizedInterpolationCollection{vdim, IPC},
    cell::AbstractCell{ref_shape},
) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, cell)^vdim
getinterpolation(
    ipc::VectorizedInterpolationCollection{vdim, IPC},
    type::Type{ref_shape},
) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, type)^vdim

"""
    NodalQuadratureRuleCollection(::InterpolationCollection)

A collection of nodal quadrature rules across different cell types.

!!! warning
    The computation for the weights is not implemented yet and hence they default to NaN.
"""
struct NodalQuadratureRuleCollection{IPC <: InterpolationCollection}
    ipc::IPC
end

function getquadraturerule(
    nqr::NodalQuadratureRuleCollection,
    cell::AbstractCell{ref_shape},
) where {ref_shape}
    ip = getinterpolation(nqr.ipc, cell)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape}([NaN for _ = 1:length(positions)], positions)
end
getquadraturerule(qrc::NodalQuadratureRuleCollection, sdh::SubDofHandler) =
    getquadraturerule(qrc, get_first_cell(sdh))


"""
    FacetQuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct FacetQuadratureRuleCollection{order} end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order}()

getquadraturerule(
    qrc::FacetQuadratureRuleCollection{order},
    cell::AbstractCell{ref_shape},
) where {order, ref_shape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(
    qrc::FacetQuadratureRuleCollection{order},
    ::Type{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler) =
    getquadraturerule(qrc, get_first_cell(sdh))


"""
    CellValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct cell values on mixed grids.
"""
struct CellValueCollection{
    QRC <: Union{QuadratureRuleCollection, NodalQuadratureRuleCollection},
    IPC <: InterpolationCollection,
}
    qrc::QRC
    ipc::IPC
end

getcellvalues(cv::CellValueCollection, cell::CellType) where {CellType <: AbstractCell} =
    CellValues(
        getquadraturerule(cv.qrc, cell),
        getinterpolation(cv.ipc, cell),
        Ferrite.geometric_interpolation(CellType),
    )
getcellvalues(qrc::CellValueCollection, sdh::SubDofHandler) =
    getcellvalues(qrc, get_first_cell(sdh))


"""
    FacetValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct facet values on mixed grids.
"""
struct FacetValueCollection{QRC <: FacetQuadratureRuleCollection, IPC <: InterpolationCollection}
    qrc::QRC
    ipc::IPC
end

getfacetvalues(fv::FacetValueCollection, cell::CellType) where {CellType <: AbstractCell} =
    FacetValues(
        getquadraturerule(fv.qrc, cell),
        getinterpolation(fv.ipc, cell),
        Ferrite.geometric_interpolation(CellType),
    )
getfacetvalues(qrc::FacetValueCollection, sdh::SubDofHandler) =
    getfacetvalues(qrc, get_first_cell(sdh))


"""
    ElementwiseData(data, offsets)

Container to handle manage quadrature data and friends on mixed grids.
"""
struct ElementwiseData{
    DataType,
    StorageType <: AbstractVector{DataType},
    IndexStorageType <: AbstractVector{<:Int},
} <: AbstractMatrix{DataType}
    data::StorageType
    offsets::IndexStorageType
    sizes::IndexStorageType
end

Base.getindex(data::ElementwiseData, i::Int) = data.data[i]
Base.length(data::ElementwiseData) = length(data.data)
Base.size(data::ElementwiseData) = (0, length(data.offsets))
function Base.show(
    io::IO,
    ::MIME"text/plain",
    data::ElementwiseData{DataType, StorageType, IndexStorageType},
) where {DataType, StorageType, IndexStorageType}
    print(
        io,
        "ElementwiseData{DataType=$DataType, StorageType=$StorageType, IndexStorageType=$IndexStorageType} with $(length(data.data)) entries and outer dimension $(length(data.offsets)).",
    )
end

function Base.setindex!(data::ElementwiseData{T}, v::T, i::Int) where {T}
    data.data[i] = v
end

function Base.getindex(data::ElementwiseData, j::Int, i::Int)
    os = data.offsets[i]:(data.offsets[i]+data.sizes[i]-1)
    dv = @view data.data[os]
    return dv[j]
end

function Base.setindex!(data::ElementwiseData{T}, v::T, j::Int, i::Int) where {T}
    os = data.offsets[i]:(data.offsets[i]+data.sizes[i]-1)
    dv = @view data.data[os]
    dv[j] = v
end


"""
    ApproximationDescriptor(symbol, interpolation_collection)
"""
struct ApproximationDescriptor
    sym::Symbol
    ipc::InterpolationCollection
end

"""
    add_subdomain!(dh, name::String, approximations::Vector{ApproximationDescriptor})
    add_subdomain!(dh, name::String, sym => interpolation_collection)
    add_subdomain!(dh, approximations)

Add the fields described by `approximations` to `dh` on the mesh's volumetric subdomain `name`, one
`SubDofHandler` per cell type occurring there. Errors if the mesh has no subdomain of that name.

The form without a name applies to the mesh's only subdomain and asserts that there is exactly one.
"""
function add_subdomain!(
    dh::DofHandler{<:Any, <:SimpleMesh},
    name::String,
    approxmations::Vector{ApproximationDescriptor},
)
    mesh = dh.grid
    cells = mesh.grid.cells
    haskey(mesh.volumetric_subdomains, name) || error(
        "Volumetric Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.volumetric_subdomains))",
    )
    for (celltype, cellset) in mesh.volumetric_subdomains[name].data
        # @info name, length(cellset)
        sdh = SubDofHandler(dh, OrderedSet{Int}([idx.idx for idx in cellset]))
        for ad in approxmations
            add!(sdh, ad.sym, getinterpolation(ad.ipc, cells[first(sdh.cellset)]))
        end
    end
end
add_subdomain!(dh, domain_name, descriptor::Pair) =
    add_subdomain!(dh, domain_name, [ApproximationDescriptor(descriptor[1], descriptor[2])])
function add_subdomain!(dh, descriptor)
    vsubdomain = get_grid(dh).volumetric_subdomains
    @assert length(vsubdomain) == 1 "Mesh has multiple subdomains. Please specify the subdomain on which the approximation is defined."
    add_subdomain!(dh, first(keys(vsubdomain)), descriptor)
end

# function add_surface_subdomain!(dh::DofHandler{<:Any, <:SimpleMesh}, name::String, approxmations::Vector{ApproximationDescriptor})
#     mesh = dh.grid
#     haskey(mesh.surface_subdomains, name) || error("Surface Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.surface_subdomains))")
#     for (celltype, cellset) in mesh.surface_subdomains[name].data
#         dh_solid_quad = SubDofHandler(dh, cellset)
#         for ad in approxmations
#             add!(dh_solid_quad, ad.sym, getinterpolation(ipc, celltype))
#         end
#     end
# end

# function add_interface_subdomain!(dh::DofHandler{<:Any, <:SimpleMesh}, name::String, approxmations::Vector{ApproximationDescriptor})
#     mesh = dh.grid
#     haskey(mesh.interface_subdomains, name) || error("Interface Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.interface_subdomains))")
#     for (celltype, cellset) in mesh.interface_subdomains[name].data
#         dh_solid_quad = SubDofHandler(dh, cellset)
#         for ad in approxmations
#             add!(dh_solid_quad, ad.sym, getinterpolation(ipc, celltype))
#         end
#     end
# end
