function Adapt.adapt_structure(to, element_cache::AnalyticalCoefficientElementCache)
    cc = Adapt.adapt_structure(to, element_cache.cc)
    nz_intervals = Adapt.adapt_structure(to, element_cache.nonzero_intervals |> cu)
    cv = element_cache.cv
    fv = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    sv = FerriteUtils.StaticCellValues(fv, gm, weights)
    return AnalyticalCoefficientElementCache(cc, nz_intervals, sv)
end

function Adapt.adapt_structure(to, coeff::AnalyticalCoefficientCache)
    f = Adapt.adapt_structure(to, coeff.f)
    coordinate_system_cache = Adapt.adapt_structure(to, coeff.coordinate_system_cache)
    return AnalyticalCoefficientCache(f, coordinate_system_cache)
end

function Adapt.adapt_structure(to, cysc::CartesianCoordinateSystemCache)
    cs = Adapt.adapt_structure(to, cysc.cs)
    cv = Adapt.adapt_structure(to, cysc.cv)
    return CartesianCoordinateSystemCache(cs, cv)
end

function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    return FerriteUtils.StaticCellValues(fv, gm, weights)
end

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

function Adapt.adapt_structure(to::Type{CUDABackend}, dh::DofHandler{sdim}) where sdim
    @show "here"
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
        dh.ndofs,
    )
    #return GPUDofHandler(dh, gpudata)
    return GPUDofHandler(gpudata)
end



function Adapt.adapt_structure(to::Type{CUDABackend}, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    @show "grid"
    node_type = typeof(first(grid.nodes))
    cells = Adapt.adapt_structure(to, grid.cells)
    nodes = Adapt.adapt_structure(to, grid.nodes)
    #TODO subdomain info
    return GPUGrid{sdim, cell_type, T, typeof(cells), typeof(nodes)}(cells, nodes)
end