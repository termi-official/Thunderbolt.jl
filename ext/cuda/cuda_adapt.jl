@adapt_structure KeFeGlobalMem
@adapt_structure FeGlobalMem
@adapt_structure KeGlobalMem

function Adapt.adapt_structure(to, element_cache::AnalyticalCoefficientElementCache)
    cc = Adapt.adapt_structure(to, element_cache.cc)
    nz_intervals = Adapt.adapt_structure(to, element_cache.nonzero_intervals |> cu)
    sv = Adapt.adapt_structure(to, element_cache.cv)
    return AnalyticalCoefficientElementCache(cc, nz_intervals, sv)
end

# TODO: not used in the current codebase
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


function Adapt.adapt_structure(strategy::CudaAssemblyStrategy, dh::DofHandler)
    IT = inttype(strategy)
    grid = _adapt(strategy, dh.grid)
    cell_dofs = dh.cell_dofs .|> (i -> convert(IT,i)) |> cu
    cell_dofs_offset = dh.cell_dofs_offset .|> (i -> convert(IT,i)) |> cu
    cell_to_sdh = dh.cell_to_subdofhandler .|> (i -> convert(IT,i)) |> cu
    subdofhandlers = dh.subdofhandlers .|> (sdh -> _adapt(strategy, sdh)) |> cu |> cudaconvert
    gpudata = DeviceDofHandlerData(
        grid,
        subdofhandlers,
        cell_dofs,
        cell_dofs_offset,
        cell_to_sdh,
        convert(IT,dh.ndofs),
    )
    return DeviceDofHandler(gpudata)
end

_symbols_to_int(symbols,IT::Type) = 1:length(symbols) .|> (sym -> convert(IT, sym))


function _adapt(strategy::CudaAssemblyStrategy, sdh::SubDofHandler)
    IT = inttype(strategy)
    cellset =  sdh.cellset |> collect .|> (x -> convert(IT, x)) |> cu |> cudaconvert
    field_names =  _symbols_to_int(sdh.field_names,IT) |> cu |> cudaconvert
    field_interpolations = sdh.field_interpolations |> convert_vec_to_concrete |> cu |> cudaconvert
    ndofs_per_cell =  sdh.ndofs_per_cell
    return DeviceSubDofHandlerData(cellset, field_names, field_interpolations, ndofs_per_cell)
end

function _adapt(strategy::CudaAssemblyStrategy, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    node_type = typeof(first(grid.nodes))
    cells =  grid.cells |> convert_vec_to_concrete  |> cu
    nodes =  grid.nodes |> cu
    #TODO subdomain info
    return DeviceGrid{sdim, cell_type, T, typeof(cells), typeof(nodes)}(cells, nodes)
end


# Adapt Coefficients #
@adapt_structure AnalyticalCoefficientCache
@adapt_structure CartesianCoordinateSystemCache

function Adapt.adapt_structure(to, cysc::FieldCoefficientCache)
    elementwise_data = Adapt.adapt_structure(to, cysc.elementwise_data |> cu)
    cv = Adapt.adapt_structure(to, cysc.cv)
    return FieldCoefficientCache(elementwise_data, cv)
end
function Adapt.adapt_structure(to, sphdf::SpatiallyHomogeneousDataField)
    timings = Adapt.adapt_structure(to, sphdf.timings |> cu)
    data = Adapt.adapt_structure(to, sphdf.data |> cu)
    return SpatiallyHomogeneousDataField(timings, data)
end