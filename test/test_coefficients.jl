@testset "Coefficient API" begin
    cell_cache = Ferrite.CellCache(generate_grid(Line, (2,)))
    qp1 = QuadraturePoint(1, Vec((0.0,)))
    qp2 = QuadraturePoint(2, Vec((0.1,)))
    ip_collection = LagrangeCollection{1}()

    @testset "ConstantCoefficient($val" for val ∈ [1.0, one(Tensor{2,2})]
        cc = ConstantCoefficient(val)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(cc, cell_cache, qp1, 0.0) ≈ val
        @test evaluate_coefficient(cc, cell_cache, qp2, 1.0) ≈ val
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(cc, cell_cache, qp1, 0.0) ≈ val
        @test evaluate_coefficient(cc, cell_cache, qp2, 1.0) ≈ val
    end

    @testset "FieldCoefficient" begin
        data_scalar = zeros(2,2)
        data_scalar[1,1] =  1.0
        data_scalar[1,2] = -1.0
        data_scalar[2,1] = -1.0
        fcs = FieldCoefficient(data_scalar, ip_collection)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(fcs, cell_cache, qp1, 0.0) ≈  0.0
        @test evaluate_coefficient(fcs, cell_cache, qp1, 1.0) ≈  0.0
        @test evaluate_coefficient(fcs, cell_cache, qp2, 0.0) ≈ -0.1
        @test evaluate_coefficient(fcs, cell_cache, qp2, 1.0) ≈ -0.1
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(fcs, cell_cache, qp1, 0.0) ≈ -0.5
        @test evaluate_coefficient(fcs, cell_cache, qp1, 1.0) ≈ -0.5
        @test evaluate_coefficient(fcs, cell_cache, qp2, 0.0) ≈ (0.1+1.0)/2.0-1.0
        @test evaluate_coefficient(fcs, cell_cache, qp2, 1.0) ≈ (0.1+1.0)/2.0-1.0

        data_vector = zeros(Vec{2,Float64},2,2)
        data_vector[1,1] = Vec((1.0,0.0))
        data_vector[1,2] = Vec((0.0,-1.0))
        data_vector[2,1] = Vec((-1.0,-0.0))
        fcv = FieldCoefficient(data_vector, ip_collection^2)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(fcv, cell_cache, qp1, 0.0) ≈ Vec((0.0,0.0))
        @test evaluate_coefficient(fcv, cell_cache, qp2, 1.0) ≈ Vec((-0.1,0.0))
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(fcv, cell_cache, qp1, 0.0) ≈ Vec((0.0,-0.5))
        @test evaluate_coefficient(fcv, cell_cache, qp2, 1.0) ≈ Vec((0.0,(0.1+1.0)/2.0-1.0))
    end

    @testset "Cartesian CoordinateSystemCoefficient" begin
        ccsc = CoordinateSystemCoefficient(CartesianCoordinateSystem(ip_collection^1))
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(ccsc, cell_cache, qp1, 0.0) ≈ Vec((-0.5,))
        @test evaluate_coefficient(ccsc, cell_cache, qp1, 1.0) ≈ Vec((-0.5,))
        @test evaluate_coefficient(ccsc, cell_cache, qp2, 0.0) ≈ Vec((-0.45,))
        @test evaluate_coefficient(ccsc, cell_cache, qp2, 1.0) ≈ Vec((-0.45,))
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(ccsc, cell_cache, qp1, 0.0) ≈ Vec((0.5,))
        @test evaluate_coefficient(ccsc, cell_cache, qp1, 1.0) ≈ Vec((0.5,))
        @test evaluate_coefficient(ccsc, cell_cache, qp2, 0.0) ≈ Vec((0.55,))
        @test evaluate_coefficient(ccsc, cell_cache, qp2, 1.0) ≈ Vec((0.55,))
    end

    @testset "AnalyticalCoefficient" begin
        ac = AnalyticalCoefficient(
            (x,t) -> norm(x)+t,
            CoordinateSystemCoefficient(CartesianCoordinateSystem(ip_collection^1))
        )
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(ac, cell_cache, qp1, 0.0) ≈  0.5
        @test evaluate_coefficient(ac, cell_cache, qp2, 0.0) ≈  0.45
        @test evaluate_coefficient(ac, cell_cache, qp1, 1.0) ≈  1.5
        @test evaluate_coefficient(ac, cell_cache, qp2, 1.0) ≈  1.45
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(ac, cell_cache, qp1, 0.0) ≈  0.5
        @test evaluate_coefficient(ac, cell_cache, qp2, 0.0) ≈  0.55
        @test evaluate_coefficient(ac, cell_cache, qp1, 1.0) ≈  1.5
        @test evaluate_coefficient(ac, cell_cache, qp2, 1.0) ≈  1.55
    end

    @testset "SpectralTensorCoefficient" begin
        eigvec = Vec((1.0,0.0))
        eigval = -1.0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(SVector((eigvec,))),
            ConstantCoefficient(SVector((eigval,))),
        )
        st = Tensor{2,2}((-1.0,0.0,0.0,0.0))
        for i in 1:2
            reinit!(cell_cache, i)
            @test evaluate_coefficient(stc, cell_cache, qp1, 0.0) ≈ st
            @test evaluate_coefficient(stc, cell_cache, qp2, 0.0) ≈ st
            @test evaluate_coefficient(stc, cell_cache, qp1, 1.0) ≈ st
            @test evaluate_coefficient(stc, cell_cache, qp2, 1.0) ≈ st
        end

        stc2 = SpectralTensorCoefficient(
            ConstantCoefficient(SVector((eigvec,))),
            ConstantCoefficient(SVector((eigval,eigval))),
        )
        @test_throws ErrorException evaluate_coefficient(stc2, cell_cache, qp1, 0.0) ≈ st
    end

    @testset "SpatiallyHomogeneousDataField" begin
        shdc = SpatiallyHomogeneousDataField(
            [1.0, 2.0],
            [Vec((0.1,)), Vec((0.2,)), Vec((0.3,))]
        )
        for i in 1:2
            reinit!(cell_cache, i)
            @test evaluate_coefficient(shdc, cell_cache, qp1, 0.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(shdc, cell_cache, qp2, 0.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(shdc, cell_cache, qp1, 1.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(shdc, cell_cache, qp2, 1.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(shdc, cell_cache, qp1, 1.1) ≈ Vec((0.2,))
            @test evaluate_coefficient(shdc, cell_cache, qp2, 1.1) ≈ Vec((0.2,))
            @test evaluate_coefficient(shdc, cell_cache, qp1, 2.0) ≈ Vec((0.2,))
            @test evaluate_coefficient(shdc, cell_cache, qp2, 2.0) ≈ Vec((0.2,))
            @test evaluate_coefficient(shdc, cell_cache, qp1, 2.1) ≈ Vec((0.3,))
            @test evaluate_coefficient(shdc, cell_cache, qp2, 2.1) ≈ Vec((0.3,))
        end
    end

    @testset "ConductivityToDiffusivityCoefficient" begin
        eigvec = Vec((1.0,0.0))
        eigval = -1.0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(SVector((eigvec,))),
            ConstantCoefficient(SVector((eigval,))),
        )
        st = Tensor{2,2}((-1.0,0.0,0.0,0.0))
        ctdc = Thunderbolt.ConductivityToDiffusivityCoefficient(
            stc,
            ConstantCoefficient(2.0),
            ConstantCoefficient(0.5),
        )
        for i in 1:2
            reinit!(cell_cache, i)
            @test evaluate_coefficient(stc, cell_cache, qp1, 0.0) ≈ st
            @test evaluate_coefficient(stc, cell_cache, qp2, 0.0) ≈ st
            @test evaluate_coefficient(stc, cell_cache, qp1, 1.0) ≈ st
            @test evaluate_coefficient(stc, cell_cache, qp2, 1.0) ≈ st
        end
    end
end
