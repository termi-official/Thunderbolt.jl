"""
"""
struct ParaViewWriter{PVD}
    filename::String
    pvd::PVD
end

ParaViewWriter(filename::String) = ParaViewWriter(filename, paraview_collection("$filename.pvd"))

function store_timestep!(io::ParaViewWriter, t, dh, u)
    mkpath(io.filename)
    vtk_grid(io.filename * "/$t.vtu", dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_save(vtk)
        io.pvd[t] = vtk
    end
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

JLD2Writer(filename::String, overwrite::Bool=true) = JLD2Writer(filename, jldopen("$filename.jld2", overwrite ? "w" : "a+"))

function store_timestep!(io::JLD2Writer, t, dh, u)
    io.fd["timesteps/$t/solution"] = u
end

function finalize!(io::JLD2Writer)
end
