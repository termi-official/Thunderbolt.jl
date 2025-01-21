@adapt_structure GlobalMemAlloc
@adapt_structure CUDACellIterator 

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

function Adapt.adapt_structure(to, dh::DofHandler{sdim}) where sdim
    grid             = adapt_structure(to, dh.grid)
    # field_names      = Tuple(sym for sym in dh.field_names)
    #IndexType        = eltype(dh.cell_dofs)
    #IndexVectorType  = CuVector{IndexType}
    cell_dofs        = adapt(to, dh.cell_dofs .|> (i -> convert(Int32,i)) |> cu) # currently you cant create Dofhandler with Int32
    cell_dofs_offset = adapt(to, dh.cell_dofs_offset .|> (i -> convert(Int32,i)) |> cu)
    cell_to_sdh      = adapt(to, dh.cell_to_subdofhandler .|> (i -> convert(Int32,i)) |> cu)
    #subdofhandlers   = Tuple(i->_convert_subdofhandler_to_gpu(cell_dofs, cell_dofs_offset, sdh) for sdh in dh.subdofhandlers)
    subdofhandlers   = adapt_structure(to,dh.subdofhandlers .|> (sdh -> Adapt.adapt_structure(to, sdh)) |> cu)
    gpudata = DeviceDofHandlerData(
        grid,
        subdofhandlers,
        # field_names,
        cell_dofs,
        cell_dofs_offset,
        cell_to_sdh,
        convert(Int32,dh.ndofs),
    )
    #return DeviceDofHandler(dh, gpudata)
    return DeviceDofHandler(gpudata)
end

_symbols_to_int32(symbols) = 1:length(symbols) .|> (sym -> convert(Int32, sym))

function Adapt.adapt_structure(to, sdh::SubDofHandler)
    cellset = Adapt.adapt_structure(to, sdh.cellset |> collect .|> (x -> convert(Int32, x)) |> cu)
    field_names = Adapt.adapt_structure(to, _symbols_to_int32(sdh.field_names) |> cu)
    field_interpolations = sdh.field_interpolations .|> (ip -> Adapt.adapt_structure(to, ip)) |> cu
    ndofs_per_cell = Adapt.adapt_structure(to, sdh.ndofs_per_cell)
    return DeviceSubDofHandlerData(cellset, field_names, field_interpolations, ndofs_per_cell)
end

function Adapt.adapt_structure(to, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    node_type = typeof(first(grid.nodes))
    cells = Adapt.adapt_structure(to, grid.cells .|> (x -> Int32.(x.nodes)) .|> eltype(grid.cells) |> cu)
    nodes = Adapt.adapt_structure(to, grid.nodes |> cu)
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