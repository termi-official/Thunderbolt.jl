using JET, Test, Tensors, Thunderbolt, StaticArrays

include("test_operators.jl")

include("test_type_stability.jl")
include("test_mesh.jl")
include("test_coefficients.jl")
include("test_microstructures.jl")

include("integration/test_contracting_cuboid.jl")
include("integration/test_waveprop_cuboid.jl")

include("test_aqua.jl")
