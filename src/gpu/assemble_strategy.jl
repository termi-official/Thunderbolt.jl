
abstract type AbstractAssemblyStrategy end

# encompass the all the required data types that needs to be worked with on the GPU
struct CudaAssemblyStrategy <: AbstractAssemblyStrategy
    floattype::Type
    inttype::Type
end

floattype(strategy::CudaAssemblyStrategy) = strategy.floattype
inttype(strategy::CudaAssemblyStrategy) = strategy.inttype

CudaDefaultAssemblyStrategy() = CudaAssemblyStrategy(Float32, Int32)