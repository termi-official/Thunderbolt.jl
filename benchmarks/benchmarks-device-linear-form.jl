using BenchmarkTools, Thunderbolt, StaticArrays,Ferrite , CUDA

# The Following is needed to enforce grid to use Float32.
left = Tensor{1, 3, Float32}((-1.0, -1.0,-1.0)) 
right = Tensor{1, 3, Float32}((1.0, 1.0, 1.0)) 

grid = generate_grid(Hexahedron , (10,10,10),left,right)

ip_collection = LagrangeCollection{1}()
qr_collection = QuadratureRuleCollection(2)
dh = DofHandler(grid)
add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
close!(dh)
cs = CartesianCoordinateSystem(grid)
protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> sin(2π * t) * exp(-norm(x)^2), CoordinateSystemCoefficient(cs)),
                [SVector((0.f0, 1.f0))]
            )



#############################
# CPU operator Benchmarking #
#############################
linop = Thunderbolt.LinearOperator(
    zeros(ndofs(dh)),
    protocol,
    qr_collection,
    dh,
)

@btime Thunderbolt.update_operator!($linop,$0.0)


#############################
# GPU operator Benchmarking #
#############################
cuda_strategy = Thunderbolt.CudaAssemblyStrategy()
cuda_op = Thunderbolt.init_linear_operator(cuda_strategy,protocol, qr_collection, dh);

## benchmark with BenchmarkTools
@btime Thunderbolt.update_operator!($cuda_op,$0.f0)

## benchmark with CUDA/Nvidia tools
CUDA.@profile Thunderbolt.update_operator!(cuda_op,0.f0)
