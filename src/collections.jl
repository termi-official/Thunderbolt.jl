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


"""
    LagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct LagrangeCollection{order} <: ScalarInterpolationCollection
end

getinterpolation(lc::LagrangeCollection{order}, cell::AbstractCell{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()
getinterpolation(lc::LagrangeCollection{order}, ::Type{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()

"""
    DiscontinuousLagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct DiscontinuousLagrangeCollection{order} <: ScalarInterpolationCollection
end

getinterpolation(lc::DiscontinuousLagrangeCollection{order}, cell::AbstractCell{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()
getinterpolation(lc::DiscontinuousLagrangeCollection{order}, ::Type{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()


"""
    VectorizedInterpolationCollection{order} <: InterpolationCollection

A collection of fixed-order vectorized Lagrange interpolations across different cell types.
"""
struct VectorizedInterpolationCollection{vdim, IPC <: ScalarInterpolationCollection} <: VectorInterpolationCollection
    base::IPC
    function VectorizedInterpolationCollection{vdim}(ip::SIPC) where {vdim, SIPC <: ScalarInterpolationCollection}
        return new{vdim, SIPC}(ip)
    end
end

Base.:(^)(ip::ScalarInterpolationCollection, vdim::Int) = VectorizedInterpolationCollection{vdim}(ip)

getinterpolation(ipc::VectorizedInterpolationCollection{vdim, IPC}, cell::AbstractCell{ref_shape}) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, cell)^vdim
getinterpolation(ipc::VectorizedInterpolationCollection{vdim, IPC}, type::Type{ref_shape}) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, type)^vdim


"""
    QuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct QuadratureRuleCollection{order}
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection{order}()

getquadraturerule(qrc::QuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = QuadratureRule{ref_shape}(order)


"""
    NodalQuadratureRuleCollection(::InterpolationCollection)

A collection of nodal quadrature rules across different cell types.

!!! warning
    The computation for the weights is not implemented yet and hence they default to NaN.
"""
struct NodalQuadratureRuleCollection{IPC <: InterpolationCollection}
    ipc::IPC
end

function getquadraturerule(nqr::NodalQuadratureRuleCollection, cell::AbstractCell{ref_shape}) where {ref_shape}
    ip = getinterpolation(nqr.ipc, cell)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape, eltype(first(positions))}([NaN for _ in 1:length(positions)], positions)
end


"""
    FacetQuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct FacetQuadratureRuleCollection{order}
end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order}()

getquadraturerule(qrc::FacetQuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = FacetQuadratureRule{ref_shape}(order)


"""
    CellValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct cell values on mixed grids.
"""
struct CellValueCollection{QRC <: Union{QuadratureRuleCollection, NodalQuadratureRuleCollection}, IPC <: InterpolationCollection}
    qrc::QRC
    ipc::IPC
end

getcellvalues(cv::CellValueCollection, cell::CellType) where {CellType <: AbstractCell} = CellValues(
    getquadraturerule(cv.qrc, cell),
    getinterpolation(cv.ipc, cell),
    Ferrite.geometric_interpolation(CellType)
)


"""
    FacetValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct face values on mixed grids.
"""
struct FacetValueCollection{QRC <: FacetQuadratureRuleCollection, IPC <: InterpolationCollection}
    qrc::QRC
    ipc::IPC
end

getfacevalues(fv::FacetValueCollection, cell::CellType) where {CellType <: AbstractCell} = FacetValues(
    getquadraturerule(fv.qrc, cell),
    getinterpolation(fv.ipc, cell),
    Ferrite.geometric_interpolation(CellType)
)
