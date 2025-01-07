# Download some assets necessary for docs/testing not stored in the repo
import Downloads

const directory = joinpath(@__DIR__, "src", "tutorials")
mkpath(directory)

for (file, url) in [
        # "spiral-wave.gif" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/spiral-wave.gif",
        # "contracting-left-ventricle.gif" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/contracting-left-ventricle.gif",
    ]
    afile = joinpath(directory, file)
    if !isfile(afile)
        Downloads.download(url, afile)
    end
end
