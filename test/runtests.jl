using JET, Test, Tensors, Thunderbolt, StaticArrays

function generate_mixed_grid_2D()
    nodes = Node.([
        Vec((-1.0,-1.0)),
        Vec(( 0.0,-1.0)),
        Vec(( 1.0,-1.0)),
        Vec((-1.0, 1.0)),
        Vec(( 0.0, 1.0)),
        Vec(( 1.0, 1.0)),
    ])
    elements = [
        Triangle(1,2,5),
        Triangle(1,5,4),
        Quadrilateral(2,3,6,5),
    ]
    cellsets = Dict((
        "Pacemaker" => OrderedSet([1])
        "Myocardium" => OrderedSet([2,3])
    ))
    return Grid(nodes, elements; cellsets)
end

include("test_operators.jl")

include("test_subdomains.jl")

include("test_solver.jl")

include("test_transfer.jl")

include("test_integrators.jl")

include("test_type_stability.jl")
include("test_mesh.jl")
include("test_coefficients.jl")
include("test_microstructures.jl")

include("integration/test_passive_structure.jl") # TODO make this a tutorial
include("integration/test_solid_mechanics.jl")
include("integration/test_waveprop_cuboid.jl")
include("integration/test_ecg.jl")

include("test_aqua.jl")
