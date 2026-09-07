# Every coefficient family evaluated at quadrature points on the device, against the same hand
# computed reference values the pre-FerriteOperators suite asserted.
#
# The evaluation runs the way a device element kernel reaches a coefficient: a Ferrite device
# `CellCache` built from the device dof handler, `reinit!`ed per cell, and `evaluate_coefficient`
# called on it with a `QuadraturePoint`. Nothing here is Thunderbolt's own device machinery -- that
# is what the assembly tests cover -- so a family that fails here fails for the assembly path too.

import Thunderbolt:
    ConductivityToDiffusivityCoefficient,
    QuadraturePoint,
    evaluate_coefficient,
    setup_coefficient_cache

@kernel function _coefficient_kernel!(vals, caches, cache, @Const(points), t, nqp)
    cellid = @index(Global, Linear)
    cc = caches[cellid]
    Ferrite.reinit!(cc, cellid)
    for q = 1:nqp
        vals[(cellid-1)*nqp+q] = evaluate_coefficient(cache, cc, QuadraturePoint(q, points[q]), t)
    end
end

"""
    device_coefficient_values(coefficient, dh, qr, t, ValueType)

`coefficient` at every quadrature point of every cell, evaluated on the device and returned on the
host, ordered cell major.
"""
function device_coefficient_values(coefficient, dh, qr, t, ::Type{VT}) where {VT}
    backend    = CUDABackend()
    sdh        = first(dh.subdofhandlers)
    ncells     = length(sdh.cellset)
    nqp        = Ferrite.getnquadpoints(qr)
    device_sdh = first(adapt(backend, dh).subdofhandlers)

    caches = Ferrite.distribute_to_workers(backend, Ferrite.CellCache(device_sdh), ncells)
    cache  = adapt(backend, setup_coefficient_cache(coefficient, qr, sdh))
    points = adapt(backend, Ferrite.getpoints(qr))
    vals   = KA.zeros(backend, VT, ncells * nqp)

    _coefficient_kernel!(backend, min(ncells, 256))(
        vals,
        caches,
        cache,
        points,
        t,
        nqp;
        ndrange = ncells,
    )
    KA.synchronize(backend)
    return Array(vals)
end

@testset "Coefficient API" begin
    left = Tensor{1, 1, Float32}((-1.0,))
    right = Tensor{1, 1, Float32}((1.0,))
    grid = generate_grid(Line, (2,), left, right)
    dh = DofHandler(grid)
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, first(grid.cells))
    add!(dh, :u, ip)
    close!(dh)
    qr  = QuadratureRule{RefLine}([1.0f0, 1.0f0], [Vec{1}((0.0f0,)), Vec{1}((0.1f0,))])
    sdh = first(dh.subdofhandlers)

    @testset "ConstantCoefficient" begin
        vals = device_coefficient_values(ConstantCoefficient(1.0f0), dh, qr, 0.0f0, Float32)
        @test vals ≈ ones(Float32, 4)
    end

    @testset "FieldCoefficient" begin
        data_scalar = zeros(Float32, 2, 2)
        data_scalar[1, 1] = 1.0f0
        data_scalar[1, 2] = -1.0f0
        data_scalar[2, 1] = -1.0f0
        vals = device_coefficient_values(
            FieldCoefficient(data_scalar, ip_collection),
            dh,
            qr,
            0.0f0,
            Float32,
        )
        @test vals ≈ [0.0f0, -0.1f0, -0.5f0, (0.1f0 + 1.0f0) / 2.0f0 - 1.0f0]

        data_vector = zeros(Vec{2, Float32}, 2, 2)
        data_vector[1, 1] = Vec((1.0f0, 0.0f0))
        data_vector[1, 2] = Vec((0.0f0, -1.0f0))
        data_vector[2, 1] = Vec((-1.0f0, -0.0f0))
        vals = device_coefficient_values(
            FieldCoefficient(data_vector, ip_collection^2),
            dh,
            qr,
            0.0f0,
            Vec{2, Float32},
        )
        @test vals ≈ [
            Vec((0.0f0, 0.0f0)),
            Vec((-0.1f0, 0.0f0)),
            Vec((0.0f0, -0.5f0)),
            Vec((0.0f0, (0.1f0 + 1.0f0) / 2.0f0 - 1.0f0)),
        ]
    end

    @testset "CartesianCoordinateSystem" begin
        vals = device_coefficient_values(
            CartesianCoordinateSystem(grid),
            dh,
            qr,
            0.0f0,
            Vec{1, Float32},
        )
        @test vals ≈ [Vec((-0.5f0,)), Vec((-0.45f0,)), Vec((0.5f0,)), Vec((0.55f0,))]
    end

    @testset "AnalyticalCoefficient" begin
        ac = AnalyticalCoefficient((x, t) -> norm(x) + t, CartesianCoordinateSystem(grid))
        @test device_coefficient_values(ac, dh, qr, 0.0f0, Float32) ≈ [0.5f0, 0.45f0, 0.5f0, 0.55f0]
        @test device_coefficient_values(ac, dh, qr, 1.0f0, Float32) ≈ [1.5f0, 1.45f0, 1.5f0, 1.55f0]
    end

    @testset "SpectralTensorCoefficient" begin
        eigvec = Vec((1.0f0, 0.0f0))
        eigval = -1.0f0
        st = Tensor{2, 2, Float32}((-1.0, 0.0, 0.0, 0.0))
        st2 = Tensor{2, 2, Float32}((-1.0, 0.0, 0.0, -1.0))

        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, 0.0f0))),
        )
        @test device_coefficient_values(stc, dh, qr, 0.0f0, Tensor{2, 2, Float32, 4}) ≈ fill(st, 4)

        stc2 = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, eigval))),
        )
        @test device_coefficient_values(stc2, dh, qr, 0.0f0, Tensor{2, 2, Float32, 4}) ≈
              fill(st2, 4)

        stc3 = SpectralTensorCoefficient(
            ConstantCoefficient(
                AnisotropicPlanarMicrostructure(Vec((1.0f0, 0.0f0)), Vec((0.0f0, 1.0f0))),
            ),
            ConstantCoefficient(SVector((eigval, eigval))),
        )
        @test device_coefficient_values(stc3, dh, qr, 0.0f0, Tensor{2, 2, Float32, 4}) ≈
              fill(st2, 4)
    end

    @testset "SpatiallyHomogeneousDataField" begin
        shdc = SpatiallyHomogeneousDataField(
            [1.0f0, 2.0f0],
            [Vec((0.1f0,)), Vec((0.2f0,)), Vec((0.3f0,))],
        )
        for (t, expected) in
            ((0.0f0, 0.1f0), (1.0f0, 0.1f0), (1.1f0, 0.2f0), (2.0f0, 0.2f0), (2.1f0, 0.3f0))
            @test device_coefficient_values(shdc, dh, qr, t, Vec{1, Float32}) ≈
                  fill(Vec((expected,)), 4)
        end
    end

    @testset "ConductivityToDiffusivityCoefficient" begin
        ctdc = ConductivityToDiffusivityCoefficient(
            SpectralTensorCoefficient(
                ConstantCoefficient(TransverselyIsotropicMicrostructure(Vec((1.0f0, 0.0f0)))),
                ConstantCoefficient(SVector((-1.0f0, 0.0f0))),
            ),
            ConstantCoefficient(2.0f0),
            ConstantCoefficient(0.5f0),
        )
        expected = fill(Tensor{2, 2, Float32}((-1.0, 0.0, 0.0, 0.0)), 4)
        @test device_coefficient_values(ctdc, dh, qr, 0.0f0, Tensor{2, 2, Float32, 4}) ≈ expected
        @test device_coefficient_values(ctdc, dh, qr, 1.0f0, Tensor{2, 2, Float32, 4}) ≈ expected
    end
end
