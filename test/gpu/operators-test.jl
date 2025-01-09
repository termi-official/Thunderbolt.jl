## TODO: Put test operators here or with cpu operators? 
using Thunderbolt
using CUDA
using Test

grid = generate_grid(Quadrilateral, (2,2))
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,1}())
close!(dh)
qrc = QuadratureRuleCollection{2}()

cs = CartesianCoordinateSystem(grid)

propertynames(dh)



protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> 1.0, CoordinateSystemCoefficient(cs)),
                [SVector((0.0, 1.0))]
            )

 



linop = Thunderbolt.LinearOperator(
    zeros(ndofs(dh)),
    protocol,
    qrc,
    dh,
)

Thunderbolt.update_operator!(linop,0.0)
@test linop.b ≈ [0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.5, 0.25, 0.25]



cuda_op = Thunderbolt.init_linear_operator(Thunderbolt.BackendCUDA,protocol, qrc, dh)
Thunderbolt.update_operator!(cuda_op,0.0)


@test Vector(cuda_op.op.b) ≈ linop.b
|