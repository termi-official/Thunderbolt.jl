"""
"""
mutable struct ParaViewWriter{PVD}
    filename::String
    pvd::PVD
    current_file::Union{WriteVTK.DatasetFile, Nothing}
end

ParaViewWriter(filename::String) = ParaViewWriter(filename, paraview_collection("$filename.pvd"), nothing)

function store_timestep!(io::ParaViewWriter, t, grid)
    if io.current_file === nothing
        mkpath(io.filename)
        io.current_file = vtk_grid(io.filename * "/$t.vtu", grid)
    end
end

function store_timestep_field!(io::ParaViewWriter, t, dh, u, sym::Symbol)
    check_subdomains(dh)
    # TODO extract symbol only
    vtk_point_data(io.current_file, dh, u, String(sym))
end

function store_timestep_field!(io::ParaViewWriter, t, dh, u, name::String)
    check_subdomains(dh)
    vtk_point_data(io.current_file, dh, u, name)
end

function store_timestep_celldata!(io::ParaViewWriter, t, u, coeff_name::String)
    if io.current_file === nothing
        mkpath(io.filename)
        io.current_file = vtk_grid(io.filename * "/$t.vtu", dh)
    end
    vtk_cell_data(io.current_file, u, coeff_name)
end

function finalize_timestep!(io::ParaViewWriter, t)
    vtk_save(io.current_file)
    io.pvd[t] = io.current_file
    io.current_file = nothing
    WriteVTK.save_file(io.pvd.xdoc, io.pvd.path)
end

function finalize!(io::ParaViewWriter)
    vtk_save(io.pvd)
end

#### Storing coefficients
function store_coefficient!(io, t, coefficient::AnisotropicPlanarMicrostructureModel, name)
    store_coefficient!(io, t, coefficient.fiber_coefficient, name*".f", t)
    store_coefficient!(io, t, coefficient.sheetlet_coefficient, name*".s", t)
end

function store_coefficient!(io, t, coefficient::OrthotropicMicrostructureModel, name)
    store_coefficient!(io, t, coefficient.fiber_coefficient, name*".f", t)
    store_coefficient!(io, t, coefficient.sheetlet_coefficient, name*".s", t)
    store_coefficient!(io, t, coefficient.normal_coefficient, name*".n", t)
end

# TODO split up compute from store
function store_coefficient!(io::ParaViewWriter, t, coefficient::FieldCoefficient, name)
    error("Unimplemented")
end

# TODO split up compute from store
function store_coefficient!(io::ParaViewWriter, t, coefficient::ConstantCoefficient{T}, name) where {T}
    data = zeros(T, getncells(grid))
    qrc = QuadratureRuleCollection(1)
    for cell_cache in CellIterator(dh)
        qr = getquadraturerule(qr_collection, getcells(get_grid(dh), cellid(cell_cache)))
        data[cellid(cell_cache)] = evaluate_coefficient(coefficient, cell_cache, first(QuadratureIterator(qr)), t)
    end
    vtk_cell_data(io.current_file, data, name)
end

# TODO split up compute from store
function store_coefficient!(io::ParaViewWriter, dh, coefficient::AnalyticalCoefficient{<:Any,<:CoordinateSystemCoefficient}, name, t::TimeType, qr_collection::QuadratureRuleCollection) where {TimeType}
    check_subdomains(dh)
    sdim = Ferrite.getdim(getcoordinateinterpolation(coefficient.cs))
    T = Base.return_types(c.f, (Vec{sdim}, TimeType)) # Extract the return type from the function
    @assert length(T) == 1 "Cannot deduce return type for analytical coefficient! Found: $T"
    _store_coefficient(T, io, dh, coefficient, name, t, qr_collection)
end

function _store_coefficient!(::Union{Type{<:Tuple{T}},Type{<:SVector{T}}}, tlen::Int, io::ParaViewWriter, dh, coefficient::AnalyticalCoefficient, t, qr_collection) where {T}
    data = zeros(T, getncells(grid), tlen)
    for cell_cache in CellIterator(dh) # TODO subdomain support
        qr = getquadraturerule(qr_collection, getcells(get_grid(dh), cellid(cell_cache)))
        for qp ∈ QuadratureIterator(qr)
            tval = evaluate_coefficient(coefficient, cell_cache, qp, t)
            for i ∈ 1:tlen
                data[cellid(cell_cache), i] += tval[I]
            end
        end
        data[cellid(cell_cache),:] ./= getnquadpoints(qr)
    end
    for i ∈ 1:tlen
        vtk_cell_data(io.current_file, data[:,i], name*".$i") # TODO component names
    end
end

function _store_coefficient!(T::Type, tlen::Int, io::ParaViewWriter, dh, coefficient::AnalyticalCoefficient, t, qr_collection)
    data = zeros(T, getncells(grid))
    for cell_cache in CellIterator(dh) # TODO subdomain support
        qr = getquadraturerule(qr_collection, getcells(get_grid(dh), cellid(cell_cache)))
        for qp ∈ QuadratureIterator(qr)
            data[cellid(cell_cache)] += evaluate_coefficient(coefficient, cell_cache, qp, t)
        end
        data[cellid(cell_cache)] /= getnquadpoints(qr)
    end
    vtk_cell_data(io.current_file, data, name)
end

function store_coefficient!(io, dh, coefficient::SpectralTensorCoefficient, name, t)
    check_subdomains(dh)
    store_coefficient!(io, dh, coefficient.eigenvalues, name*".λ", t) # FIXME. PLS...
    store_coefficient!(io, dh, coefficient.eigenvectors, name*".ev.", t)
end

# TODO split up compute from store
# TODO revisit if this might be expressed as some cofficient which is then stored (likely FieldCoefficient) - I think this is basically similar to `interpolate_gradient_field` in FerriteViz
function store_green_lagrange!(io::ParaViewWriter, dh, u::AbstractVector, a_coeff, b_coeff, cv, name, t)
    check_subdomains(dh)
    # TODO subdomain support
    field_idx = find_field(dh, :displacement) # TODO abstraction layer
    for cell_cache ∈ CellIterator(dh)
        reinit!(cv, cell_cache)
        global_dofs = celldofs(cell_cache)
        field_dofs  = dof_range(sdh, field_idx)
        uₑ = @view u[global_dofs] # element dofs
        for qp in QuadratureIterator(cv)
            ∇u = function_gradient(cv, qp, uₑ)

            F = one(∇u) + ∇u
            C = tdot(F)
            E = (C-one(C))/2.0

            a = evaluate_coefficient(a_coeff, cell_cache, qp, time)
            b = evaluate_coefficient(b_coeff, cell_cache, qp, time)

            E[cellid(cell)] += a ⋅ E ⋅ b
        end
        E[cellid(cell)] /= getnquadpoints(cv)
    end
    vtk_cell_data(io.current_file, E, name)
end

"""
"""
struct JLD2Writer{FD}
    filename::String
    fd::FD
end

JLD2Writer(filename::String, overwrite::Bool=true) = JLD2Writer(filename, jldopen("$filename.jld2", overwrite ? "a+" : "w"; compress = true))

function store_timestep!(io::JLD2Writer, t, grid)
    t ≈ 0.0 && (io.fd["grid"] = grid)
    io.fd["timesteps/$t/solution"] = u
end

function store_timestep_field!(io::JLD2Writer, t, dh, u, coeff_name::String)
    check_subdomains(dh)
    t ≈ 0.0 && (io.fd["dh-$coeff_name"] = dh)
    io.fd["timesteps/$t/coefficient/$coeff_name"] = u
end

function store_timestep_celldata!(io::JLD2Writer, t, u, coeff_name::String)
    check_subdomains(dh)
    io.fd["timesteps/$t/celldata/$coeff_name"] = u
end

finalize_timestep!(io::JLD2Writer, t) = nothing
finalize!(io::JLD2Writer) = nothing

function reorder_nodal!(dh::DofHandler)
    @assert length(dh.field_names) == 1 "Just single field possible."
    grid = Ferrite.get_grid(dh)
    for sdh in dh.subdofhandlers
        firstcellidx = first(sdh.cellset)
        celltype = typeof(getcells(grid,firstcellidx))
        @assert sdh.field_interpolations[1] == Ferrite.default_interpolation(celltype)
        for i ∈ 1:getncells(grid)
            dh.cell_dofs[dh.cell_dofs_offset[i]:(dh.cell_dofs_offset[i]+Ferrite.ndofs_per_cell(dh, i)-1)] .= getcells(grid,i).nodes
        end
    end
end

function to_ferrite_elements(cells_vtk::Vector{MeshCell{VTKCellType, Vector{Int64}}})
    celltype = if cells_vtk[1].ctype == VTKCellTypes.VTK_TETRA
        Tetrahedron
    elseif cells_vtk[1].ctype == VTKCellTypes.VTK_HEXAHEDRON
        Hexahedron
    else 
        @error "Unknown cell type" cells_vtk[1].ctype
    end
    cells_ferrite = Vector{celltype}(undef, length(cells_vtk))
    for (i,cell_vtk) in enumerate(cells_vtk)
        cells_ferrite[i] = if cells_vtk[1].ctype == VTKCellTypes.VTK_TETRA
            Tetrahedron(ntuple(i->cell_vtk.connectivity[i],4))
        elseif cells_vtk[1].ctype == VTKCellTypes.VTK_HEXAHEDRON
            Hexahedron(ntuple(i->cell_vtk.connectivity[i],8))
        end
    end
    return cells_ferrite
end

function read_vtk_cobivec(filename::String, transmural_id::String, apicobasal_id::String, radial_id::String, transventricular_id::String)
    vtk = VTKFile(filename)
    points_vtk = get_points(vtk)
    points_ferrite = [Vec{3}(point) for point in eachcol(points_vtk)]
    cells_vtk = to_meshcells(get_cells(vtk))
    cells_ferrite = to_ferrite_elements(cells_vtk)

    grid = Grid(cells_ferrite, Node.(points_ferrite))

    gip =  Ferrite.default_interpolation(typeof(grid.cells[1]))

    dh = DofHandler(grid)
    add!(dh, :u, gip)
    close!(dh)
    reorder_nodal!(dh)

    all_data = get_point_data(vtk)
    u_transmural = get_data(all_data[transmural_id])
    u_apicobasal = get_data(all_data[apicobasal_id])
    u_radial = get_data(all_data[radial_id])
    u_transventricular = get_data(all_data[transventricular_id])

    epicardium = Set{FaceIndex}()
    endocardium = Set{FaceIndex}()
    for (cellidx,cell) ∈ enumerate(grid.cells)
        for (faceidx,facenodes) in enumerate(Ferrite.faces(cell))
            # facenodes = cell.nodes[collect(face)]
            if all(u_transmural[collect(facenodes)] .> 1.0-1e-6)
                push!(endocardium, FaceIndex(cellidx, faceidx))
            end
            if all(u_transmural[collect(facenodes)] .< 1e-6)
                push!(epicardium, FaceIndex(cellidx, faceidx))
            end
        end
    end
    Ferrite.addfaceset!(grid, "Epicardium", epicardium)
    Ferrite.addfaceset!(grid, "Endocardium", endocardium)

    return BiVCoordinateSystem(
        dh,
        collect(u_transmural),
        collect(u_apicobasal),
        collect(u_radial),
        collect(u_transventricular)
    )
end
