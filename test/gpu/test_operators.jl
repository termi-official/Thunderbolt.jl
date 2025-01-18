left = Tensor{1, 2, Float32}((-1.0, -1.0)) # define the left bottom corner of the grid.
right = Tensor{1, 2, Float32}((1.0, 1.0)) # define the right top corner of the grid.
grid = generate_grid(Quadrilateral, (2,2),left,right)
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,1}())
close!(dh)
qrc = QuadratureRuleCollection{2}()

cs = CartesianCoordinateSystem(grid)

propertynames(dh)



protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> 1.f0, CoordinateSystemCoefficient(cs)),
                [SVector((0.f0, 1.f0))]
            )

 



linop = Thunderbolt.LinearOperator(
    zeros(ndofs(dh)),
    protocol,
    qrc,
    dh,
)

Thunderbolt.update_operator!(linop,0.0)
@test linop.b ≈ [0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.5, 0.25, 0.25]



cuda_op = Thunderbolt.init_linear_operator(CUDABackend,protocol, qrc, dh);
Thunderbolt.update_operator!(cuda_op,0.0)


@test Vector(cuda_op.op.b) ≈ linop.b


