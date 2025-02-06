###################
## adapt Buffers ##
###################

@adapt_structure KeFeGlobalMem
@adapt_structure FeGlobalMem
@adapt_structure KeGlobalMem

#####################################
## Shallow adaption for DofHandler ##
#####################################
function Adapt.adapt_structure(strategy::CudaAssemblyStrategy, dh::DofHandler)
    IT = inttype(strategy)
    grid = _adapt(strategy, dh.grid)
    cell_dofs = dh.cell_dofs .|> (i -> convert(IT,i)) |> cu
    cell_dofs_offset = dh.cell_dofs_offset .|> (i -> convert(IT,i)) |> cu
    cell_to_sdh = dh.cell_to_subdofhandler .|> (i -> convert(IT,i)) |> cu
    subdofhandlers = dh.subdofhandlers .|> (sdh -> _adapt(strategy, sdh)) 
    gpudata = DeviceDofHandlerData(
        grid,
        subdofhandlers,
        cell_dofs,
        cell_dofs_offset,
        cell_to_sdh,
        convert(IT,dh.ndofs),
    )
    return DeviceDofHandler(dh,gpudata)
end

_symbols_to_int(symbols,IT::Type) = 1:length(symbols) .|> (sym -> convert(IT, sym))


function _adapt(strategy::CudaAssemblyStrategy, sdh::SubDofHandler)
    IT = inttype(strategy)
    cellset =  sdh.cellset |> collect .|> (x -> convert(IT, x)) |> cu 
    field_names =  _symbols_to_int(sdh.field_names,IT) |> cu 
    field_interpolations = sdh.field_interpolations |> convert_vec_to_concrete |> cu 
    ndofs_per_cell =  sdh.ndofs_per_cell
    return DeviceSubDofHandlerData(cellset, field_names, field_interpolations, ndofs_per_cell)
end

function _adapt(::CudaAssemblyStrategy, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    node_type = typeof(first(grid.nodes))
    cells =  grid.cells |> convert_vec_to_concrete  |> cu
    nodes =  grid.nodes |> cu
    #TODO subdomain info
    return DeviceGrid{sdim, cell_type, T, typeof(cells), typeof(nodes)}(cells, nodes)
end

########################################
## Deep adaption for DeviceDofHandler ##
########################################
function Thunderbolt.deep_adapt(strategy::CudaAssemblyStrategy, dh::DeviceDofHandlerData)
    # here we need to perform deep adaption
    grid = dh.grid
    cell_dofs = dh.cell_dofs
    cell_dofs_offset = dh.cell_dofs_offset 
    cell_to_sdh = dh.cell_to_subdofhandler
    subdofhandlers = dh.subdofhandlers .|> (sdh -> _deep_adapt(strategy, sdh)) |> cu
    ndofs = dh.ndofs
    device_dh = DeviceDofHandlerData(
        grid,
        subdofhandlers,
        cell_dofs,
        cell_dofs_offset,
        cell_to_sdh,
        ndofs,
    )
    return device_dh
end

_symbols_to_int(symbols,IT::Type) = 1:length(symbols) .|> (sym -> convert(IT, sym))


function _deep_adapt(::CudaAssemblyStrategy, sdh::DeviceSubDofHandlerData)
    # deep adaption
    cellset =  sdh.cellset  |> cudaconvert
    field_names =  sdh.field_names |> cudaconvert
    field_interpolations = sdh.field_interpolations  |> cudaconvert
    ndofs_per_cell =  sdh.ndofs_per_cell
    return DeviceSubDofHandlerData(cellset, field_names, field_interpolations, ndofs_per_cell)
end


######################
## adapt Coefficients ##
######################
@adapt_structure AnalyticalCoefficientCache
@adapt_structure CartesianCoordinateSystemCache
@adapt_structure FieldCoefficientCache
@adapt_structure AnalyticalCoefficientElementCache
@adapt_structure SpatiallyHomogeneousDataField


function Adapt.adapt_structure(::CudaAssemblyStrategy, element_cache::AnalyticalCoefficientElementCache)
    cc = adapt_structure(CuArray, element_cache.cc)
    nz_intervals = adapt(CuArray, element_cache.nonzero_intervals )
    sv = adapt_structure(CuArray, element_cache.cv)
    return AnalyticalCoefficientElementCache(cc, nz_intervals, sv)
end

function Adapt.adapt_structure(::CudaAssemblyStrategy, cysc::FieldCoefficientCache)
    elementwise_data = adapt(CuArray, cysc.elementwise_data)
    cv = adapt_structure(CuArray, cysc.cv)
    return FieldCoefficientCache(elementwise_data, cv)
end
function Adapt.adapt_structure(::CudaAssemblyStrategy, sphdf::SpatiallyHomogeneousDataField)
    timings = adapt(CuArray, sphdf.timings )
    data = adapt(CuArray, sphdf.data )
    return SpatiallyHomogeneousDataField(timings, data)
end