using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    @timeit dto "Revise.revise()" Revise.revise()
end

using Documenter, DocumenterCitations, Thunderbolt

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# Generate tutorials and how-to guides
include("generate.jl")

bibtex_plugin = CitationBibliography(
    joinpath(@__DIR__, "src", "assets", "references.bib"),
    style=:numeric
)

# Build documentation.
@timeit dto "makedocs" makedocs(
    format = Documenter.HTML(
        assets = [
            "assets/custom.css",
            "assets/citations.css",
            # "assets/favicon.ico"
        ],
        # canonical = "https://localhost/",
        collapselevel = 1,
    ),
    sitename = "Thunderbolt.jl",
    doctest = false,
    warnonly = true,
    draft = liveserver,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "Continuum Mechanics" => [
                "CM01: Simple Active Stress" => "tutorials/mechanics_simple-active-stress.md",
            ],
            "Electrophysiology" => [
                "EP01: Simple Active Stress" => "tutorials/mechanics_simple-active-stress.md",
            ],
        ],
        "Topic Guides" => [
            "Overview" => "topics/index.md",
            "topics/operators.md",
            "topics/couplers.md",
            "topics/time-integration.md",
        ],
        "How-to guides" => [
            "Overview" => "howto/index.md",
            "howto/benchmarking.md",
        ],
        "API Reference" => [
            "Overview" => "api-reference/index.md",
            "api-reference/models.md",
            "api-reference/mesh.md",
            "api-reference/functions.md",
            "api-reference/problems.md",
            "api-reference/discretization.md",
            "api-reference/operators.md",
            "api-reference/solver.md",
            "api-reference/utility.md",
        ],
        "Developer Documentation" => [
            "Overview" => "devdocs/index.md",
            "devdocs/element_interface.md",
            "devdocs/domain_management.md",
        ],
        "references.md",
        ],
    plugins = [
        bibtex_plugin,
    ]
)

# make sure there are no *.vtu files left around from the build
@timeit dto "remove vtk files" cd(joinpath(@__DIR__, "build", "tutorials")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end

# Deploy built documentation
if !liveserver
    @timeit dto "deploydocs" deploydocs(
        repo = "github.com/termi-official/Thunderbolt.jl.git",
        push_preview=true,
        versions = [
            "stable" => "v^",
            "dev" => "dev"
        ]
    )
end

print_timer(dto)
