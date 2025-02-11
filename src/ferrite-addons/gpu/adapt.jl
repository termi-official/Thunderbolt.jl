############
# adapt.jl #
############
import Tensors: Vec
Adapt.@adapt_structure QuadratureValuesIterator
Adapt.@adapt_structure StaticQuadratureValues
Adapt.@adapt_structure DeviceCellIterator 
Adapt.@adapt_structure DeviceGrid
Adapt.@adapt_structure DeviceDofHandler
Adapt.@adapt_structure DeviceDofHandlerData
Adapt.@adapt_structure DeviceSubDofHandlerData
#Adapt.@adapt_structure QuadraturePoint
Adapt.@adapt_structure Vec

function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    #qps = Adapt.adapt(to,ntuple(i -> Adapt.adapt_structure(to,QuadraturePoint(i, cv.qr.points[i])), n_quadoints))
    qps = Adapt.adapt(to,ntuple(i -> QuadraturePoint(i, Adapt.adapt_structure(to,Vec{1}((0.0f0,)))), n_quadoints))
    return StaticCellValues(fv, gm, weights,qps)
end

function Adapt.adapt_structure(to,qp::QuadraturePoint)
    @show "A7a"
    i = Adapt.adapt(to, qp.i)
    ξ = Adapt.adapt_structure(to, qp.ξ)
    return QuadraturePoint(i, ξ)
end

function Adapt.adapt_structure(to,qp::Vec{dim,T}) where {dim,T}
    @show "A7a2"
    return Vec{dim,T}(Adapt.adapt(to, qp.data))
end
