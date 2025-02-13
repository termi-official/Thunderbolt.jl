############
# adapt.jl #
############
Adapt.@adapt_structure QuadratureValuesIterator
Adapt.@adapt_structure StaticQuadratureValues
Adapt.@adapt_structure DeviceCellIterator 
Adapt.@adapt_structure DeviceGrid
Adapt.@adapt_structure DeviceDofHandler
Adapt.@adapt_structure DeviceDofHandlerData
Adapt.@adapt_structure DeviceSubDofHandler

function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    positions = Adapt.adapt(to,ntuple(i -> Adapt.adapt(to,cv.qr.points[i].data), n_quadoints))
    return StaticCellValues(fv, gm,weights,positions)
end
