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
    QuadratureRuleCollection(order)

A collection of quadrature rules across different cell types.
"""
struct QuadratureRuleCollection{order}
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection{order}()

getquadraturerule(qrc::QuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = QuadratureRule{ref_shape}(order)


"""
    FaceQuadratureRuleCollection(order)

A collection of quadrature rules across different cell types.
"""
struct FaceQuadratureRuleCollection{order}
end

FaceQuadratureRuleCollection(order::Int) = FaceQuadratureRuleCollection{order}()

getquadraturerule(qrc::FaceQuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = FaceQuadratureRule{ref_shape}(order)


"""
    CellValueCollection

Helper to construct and query the correct cell values on mixed grids.
"""
struct CellValueCollection{QRC,IPC,IPGC}
    qrc::QRC
    ipc::IPC
    ipgc::IPGC
end

CellValueCollection(qr:: QuadratureRuleCollection, ip::InterpolationCollection) = CellValueCollection(qr,ip,ip)

getcellvalues(cv::CellValueCollection, cell::AbstractCell{ref_shape}) where {ref_shape} = CellValues(
    getquadraturerule(cv.qrc, cell),
    getinterpolation(cv.ipc, cell),
    getinterpolation(cv.ipgc, cell)
)

"""
    FaceValueCollection

Helper to construct and query the correct face values on mixed grids.
"""
struct FaceValueCollection{QRC,IPC,IPGC}
    qrc::QRC
    ipc::IPC
    ipgc::IPGC
end

getfacevalues(fv::FaceValueCollection, cell::AbstractCell{ref_shape}) where {ref_shape} = FaceValues(
    getquadraturerule(fv.qrc, cell),
    getinterpolation(fv.ipc, cell),
    getinterpolation(fv.ipgc, cell)
)
