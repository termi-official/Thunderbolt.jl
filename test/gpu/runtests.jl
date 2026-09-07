using Test
using CUDA

# Everything below needs a device. Without one the suite reports that it did nothing rather than
# failing, so it can be included unconditionally by a runner that does not know the machine.
if !CUDA.functional()
    @warn "CUDA is not functional here -- skipping the Thunderbolt GPU test suite."
else
    using Thunderbolt
    using LinearSolve
    using LinearAlgebra
    using OrdinaryDiffEqOperatorSplitting
    using StaticArrays

    import Thunderbolt: num_states, solution_size, ThreadedSparseMatrixCSR

    @testset "Thunderbolt GPU" begin
        include("test_pointwise.jl")
        include("test_diffusion.jl")
        include("test_split.jl")
    end
end

# `test_operators.jl` and `test_coefficients.jl` are deliberately not included: both are written
# against the device assembly surface that the FerriteOperators transition removed, and both wait on
# the FerriteOperators GPU device slice. See the TODO at the top of each.
