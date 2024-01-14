"""
    QuadraturePoint{dim, T}

A simple helper to carry quadrature point information.
"""
struct QuadraturePoint{dim, T}
    i::Int
    ξ::Vec{dim, T}
end

"""
    QuadratureIterator(::QuadratureRule)
    QuadratureIterator(::FaceQuadratureRule, local_face_idx::Int)
    QuadratureIterator(::CellValues)
    QuadratureIterator(::FaceValues)

A helper to loop over the quadrature points in some rule or cache with type [`QuadraturePoint`](@ref).
"""
struct QuadratureIterator{QR<:QuadratureRule}
    qr::QR
end
QuadratureIterator(fqr::FaceQuadratureRule, local_face_idx::Int) = QuadratureIterator(fqr.face_rules[local_face_idx])
QuadratureIterator(cv::CellValues) = QuadratureIterator(cv.qr)
QuadratureIterator(fv::FaceValues) = QuadratureIterator(fv.fqr.face_rules[fv.current_face[]])

function Base.iterate(iterator::QuadratureIterator, i = 1)
    i > getnquadpoints(iterator.qr) && return nothing
    return (QuadraturePoint(i,Ferrite.getpoints(iterator.qr)[i]), i + 1)
end
Base.eltype(::Type{<:QuadraturePoint{<:QuadratureRule{<:AbstractRefShape, dim, T}}}) where {dim, T} = QuadraturePoint{dim, T}
Base.length(iterator::QuadratureIterator) = length(Ferrite.getnquadpoints(iterator.qr))

Ferrite.spatial_coordinate(fe_v::Ferrite.AbstractValues, qp::QuadraturePoint, x::AbstractVector{<:Vec}) = Ferrite.spatial_coordinate(fe_v, qp.i, x)

Ferrite.getdetJdV(cv::CellValues, qp::QuadraturePoint) = Ferrite.getdetJdV(cv, qp.i)
Ferrite.shape_value(cv::CellValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_value(cv, qp.i, base_fun_idx)
Ferrite.shape_gradient(cv::CellValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_gradient(cv, qp.i, base_fun_idx)
Ferrite.function_value(cv::CellValues, qp::QuadraturePoint, ue) = Ferrite.function_value(cv, qp.i, ue)
Ferrite.function_gradient(cv::CellValues, qp::QuadraturePoint, ue) = Ferrite.function_gradient(cv, qp.i, ue)

Ferrite.getdetJdV(fv::FaceValues, qp::QuadraturePoint) = Ferrite.getdetJdV(fv, qp.i)
Ferrite.shape_value(fv::FaceValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_value(fv, qp.i, base_fun_idx)
Ferrite.shape_gradient(fv::FaceValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_gradient(fv, qp.i, base_fun_idx)
Ferrite.function_value(fv::FaceValues, qp::QuadraturePoint, ue) = Ferrite.function_value(fv, qp.i, ue)
Ferrite.function_gradient(fv::FaceValues, qp::QuadraturePoint, ue) = Ferrite.function_gradient(fv, qp.i, ue)
Ferrite.getnormal(fv::FaceValues, qp::QuadraturePoint) = Ferrite.getnormal(fv, qp.i)
