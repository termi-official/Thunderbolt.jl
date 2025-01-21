#using CUDA # Throws error (LoadError: UndefVarError: `backend` not defined in `GPUArrays`)
using Thunderbolt
using Test
using StaticArrays

include("test_operators.jl")
include("test_coefficients.jl")