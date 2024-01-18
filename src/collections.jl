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
    QuadratureRuleCollection

A collection of quadrature rules across different cell types.
"""
struct QuadratureRuleCollection{QRW, QRH, QRT}
    qr_wedge::QRW
    qr_hex::QRH
    qr_tet::QRT
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection(
    QuadratureRule{RefPrism}(order), 
    QuadratureRule{RefHexahedron}(order), 
    QuadratureRule{RefTetrahedron}(order)
)

getquadraturerule(qrc::QuadratureRuleCollection{QRW, QRH, QRT}, cell::Hexahedron) where {QRW, QRH, QRT} = qrc.qr_hex
getquadraturerule(qrc::QuadratureRuleCollection{QRW, QRH, QRT}, cell::Wedge) where {QRW, QRH, QRT} = qrc.qr_wedge
getquadraturerule(qrc::QuadratureRuleCollection{QRW, QRH, QRT}, cell::Tetrahedron) where {QRW, QRH, QRT} = qrc.qr_tet

getquadraturerule(qrc::QuadratureRuleCollection{QRW, QRH, QRT}, cell::Type{Hexahedron}) where {QRW, QRH, QRT} = qrc.qr_hex
getquadraturerule(qrc::QuadratureRuleCollection{QRW, QRH, QRT}, cell::Type{Wedge}) where {QRW, QRH, QRT} = qrc.qr_wedge
getquadraturerule(qrc::QuadratureRuleCollection{QRW, QRH, QRT}, cell::Type{Tetrahedron}) where {QRW, QRH, QRT} = qrc.qr_tet

"""
    CellValueCollection

Helper to construct and query the correct cell values on mixed grids.
"""
struct CellValueCollection{CVW, CVH, CVT}
    cv_wedge::CVW
    cv_hex::CVH
    cv_tet::CVT
end

CellValueCollection(qr:: QuadratureRuleCollection, ip::InterpolationCollection, ip_geo::InterpolationCollection = ip) = CellValueCollection(
    CellValues(getquadraturerule(qr, Wedge), getinterpolation(ip, RefPrism), getinterpolation(ip_geo, RefPrism)), 
    CellValues(getquadraturerule(qr, Hexahedron), getinterpolation(ip, RefHexahedron), getinterpolation(ip_geo, RefHexahedron)), 
    CellValues(getquadraturerule(qr, Tetrahedron), getinterpolation(ip, RefTetrahedron), getinterpolation(ip_geo, RefTetrahedron)), 
)

getcellvalues(cv::CellValueCollection{CVW, CVH, CVT}, cell::Hexahedron) where {CVW, CVH, CVT} = cv.cv_hex
getcellvalues(cv::CellValueCollection{CVW, CVH, CVT}, cell::Wedge) where {CVW, CVH, CVT} = cv.cv_wedge
getcellvalues(cv::CellValueCollection{CVW, CVH, CVT}, cell::Tetrahedron) where {CVW, CVH, CVT} = cv.cv_tet

# getcellvalues(cv::CellValueCollection{CVW, CVH, CVT}, cell::Type{Hexahedron}) where {CVW, CVH, CVT} = cv.cv_hex
# getcellvalues(cv::CellValueCollection{CVW, CVH, CVT}, cell::Type{Wedge}) where {CVW, CVH, CVT} = cv.cv_wedge
# getcellvalues(cv::CellValueCollection{CVW, CVH, CVT}, cell::Type{Tetrahedron}) where {CVW, CVH, CVT} = cv.cv_tet

"""
    FaceValueCollection

Helper to construct and query the correct face values on mixed grids.
"""
struct FaceValueCollection{CVW, CVH, CVT}
    cv_wedge::CVW
    cv_hex::CVH
    cv_tet::CVT
end

getcellvalues(cv::FaceValueCollection{CVW, CVH, CVT}, cell::Hexahedron) where {CVW, CVH, CVT} = cv.cv_hex
getcellvalues(cv::FaceValueCollection{CVW, CVH, CVT}, cell::Wedge) where {CVW, CVH, CVT} = cv.cv_wedge
getcellvalues(cv::FaceValueCollection{CVW, CVH, CVT}, cell::Tetrahedron) where {CVW, CVH, CVT} = cv.cv_tet
