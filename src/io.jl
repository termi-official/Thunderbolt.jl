"""
"""
mutable struct ParaViewWriter{PVD}
    filename::String
    pvd::PVD
    current_file::Union{WriteVTK.DatasetFile, Nothing}
end

ParaViewWriter(filename::String) = ParaViewWriter(filename, paraview_collection("$filename.pvd"), nothing)

function store_timestep!(io::ParaViewWriter, t, dh, u)
    if io.current_file === nothing
        mkpath(io.filename)
        io.current_file = vtk_grid(io.filename * "/$t.vtu", dh)
    end
    vtk_point_data(io.current_file, dh, u)
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
end

function finalize!(io::ParaViewWriter)
    vtk_save(io.pvd)
end

"""
"""
struct JLD2Writer{FD}
    filename::String
    fd::FD
end

JLD2Writer(filename::String, overwrite::Bool=true) = JLD2Writer(filename, jldopen("$filename.jld2", overwrite ? "a+" : "w"; compress = true))

function store_timestep!(io::JLD2Writer, t, dh, u)
    t ≈ 0.0 && (io.fd["dh"] = dh)
    io.fd["timesteps/$t/solution"] = u
end

function store_timestep_field!(io::JLD2Writer, t, dh, u, coeff_name::String)
    t ≈ 0.0 && (io.fd["dh-$coeff_name"] = dh)
    io.fd["timesteps/$t/coefficient/$coeff_name"] = u
end

function store_timestep_celldata!(io::JLD2Writer, t, u, coeff_name::String)
    io.fd["timesteps/$t/celldata/$coeff_name"] = u
end

finalize_timestep!(io::JLD2Writer, t) = nothing
finalize!(io::JLD2Writer) = nothing
