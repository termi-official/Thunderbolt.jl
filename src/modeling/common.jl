# Common modeling primitives are found here

# TODO actually use this :)
abstract type SteadyStateInternalVariable end

include("core/coefficients.jl")

include("core/boundary_conditions.jl")

abstract type AbstractBilinearIntegrator end
include("core/mass.jl")
include("core/diffusion.jl")
