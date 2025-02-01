function coeffs_kernel!(Vals,dh_,coeff_cache,cv,t)
    dh = dh_.gpudata
    for cell in CellIterator(dh, convert(Int32,1))
        cell_id = cellid(cell)
        coords = getcoordinates(cell)
        for (i,qv) in pairs(QuadratureValuesIterator(cv, coords))
            n_basefuncs = getnbasefunctions(cv)
            fx =  evaluate_coefficient(coeff_cache, cell, qv, t)
            Vals[(cell_id-1)*n_basefuncs + i] = fx
        end
    end
    return nothing
end


@testset "Coefficient API" begin
    import Thunderbolt: setup_coefficient_cache, evaluate_coefficient,ConductivityToDiffusivityCoefficient
    import Thunderbolt.FerriteUtils:
        CellIterator, QuadratureValuesIterator, cellid, getcoordinates, getnbasefunctions
    import CUDA: @cuda
    import Adapt: adapt_structure 


    left = Tensor{1, 1, Float32}((-1.0,)) # define the left bottom corner of the grid.
    right = Tensor{1, 1, Float32}((1.0,)) # define the right top corner of the grid.
    grid = generate_grid(Line, (2,),left,right)
    dh = DofHandler(grid)
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, first(grid.cells))
    add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
    close!(dh)
    qr  = QuadratureRule{RefLine}([1.f0, 1.f0], [Vec{1}((0.f0,)), Vec{1}((0.1f0,))])
    cellvalues = CellValues(Float32, qr, ip)
    n_cells = grid.cells |> length
    sdh = first(dh.subdofhandlers)

    cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

    @testset "ConstantCoefficient($val" for val ∈ [1.f0]
        cc = ConstantCoefficient(val)
        coeff_cache = setup_coefficient_cache(cc, qr, sdh)
        correct_vals = ones(Float32, n_cells * getnbasefunctions(cellvalues))
        Vals = zeros(Float32, n_cells * getnbasefunctions(cellvalues)) |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
    end

    @testset "FieldCoefficient" begin
        data_scalar = zeros(Float32,2,2)
        data_scalar[1,1] =  1.f0
        data_scalar[1,2] = -1.f0
        data_scalar[2,1] = -1.f0
        fcs = FieldCoefficient(data_scalar, ip_collection)
        coeff_cache = setup_coefficient_cache(fcs, qr, sdh)
        correct_vals = [0.f0,-0.1f0,-0.5f0,(0.1f0+1.0f0)/2.0f0-1.0f0]
        Vals = zeros(Float32, correct_vals |> length) |> cu
        CUDA.@cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
        
        data_vector = zeros(Vec{2,Float32},2,2)
        data_vector[1,1] = Vec((1.f0,0.f0))
        data_vector[1,2] = Vec((0.f0,-1.f0))
        data_vector[2,1] = Vec((-1.f0,-0.f0))
        fcv = FieldCoefficient(data_vector, ip_collection^2)
        coeff_cache = setup_coefficient_cache(fcv, qr, sdh)
        correct_vals = [Vec((0.f0,0.f0)), Vec((-0.1f0,0.f0)),Vec((0.f0,-0.5f0)),Vec((0.f0,(0.1f0+1.f0)/2.f0-1.f0))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
    end
   
    @testset "Cartesian CoordinateSystemCoefficient" begin
        ccsc = CoordinateSystemCoefficient(CartesianCoordinateSystem(grid))
        coeff_cache = setup_coefficient_cache(ccsc, qr, sdh)
        correct_vals = [Vec((-0.5f0,)),Vec((-0.45f0,)),Vec((0.5f0,)),Vec((0.55f0,))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
    end

    @testset "AnalyticalCoefficient" begin
        ac = AnalyticalCoefficient(
            (x,t) -> norm(x)+t,
            CoordinateSystemCoefficient(CartesianCoordinateSystem(grid))
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(ac, qr, sdh)
        correct_vals = [0.5f0,0.45f0,0.5f0,0.55f0]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
        correct_vals = [1.5f0,1.45f0,1.5f0,1.55f0]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,1.f0)
        @test Vector(Vals) ≈ correct_vals
    end

    @testset "SpectralTensorCoefficient" begin
        eigvec = Vec((1.f0,0.f0))
        eigval = -1.f0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval,0.f0))),
        )
        st = Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0))
        coeff_cache = setup_coefficient_cache(stc, qr, sdh)
        correct_vals = [st,st,st,st]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals

        st2 = Tensor{2,2,Float32}((-1.0,0.0,0.0,-1.0))
        stc2 = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval,eigval))),
        )
        coeff_cache = setup_coefficient_cache(stc2, qr, sdh)
        correct_vals = [st2,st2,st2,st2]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals

        stc3 = SpectralTensorCoefficient(
            ConstantCoefficient(AnisotropicPlanarMicrostructure(Vec((1.f0,0.f0)), Vec((0.f0,1.f0)))),
            ConstantCoefficient(SVector((eigval,eigval))),
        )
        coeff_cache = setup_coefficient_cache(stc3, qr, sdh)
        correct_vals = [st2,st2,st2,st2]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
    end

    @testset "SpatiallyHomogeneousDataField" begin
        shdc = SpatiallyHomogeneousDataField(
            [1.f0, 2.f0],
            [Vec((0.1f0,)), Vec((0.2f0,)), Vec((0.3f0,))]
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(shdc, qr, sdh)
        correct_vals = [Vec((0.1f0,)),Vec((0.1f0,)),Vec((0.1f0,)),Vec((0.1f0,))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
        correct_vals = [Vec((0.1f0,)),Vec((0.1f0,)),Vec((0.1f0,)),Vec((0.1f0,))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,1.f0)
        @test Vector(Vals) ≈ correct_vals
        correct_vals = [Vec((0.2f0,)),Vec((0.2f0,)),Vec((0.2f0,)),Vec((0.2f0,))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,1.1f0)
        @test Vector(Vals) ≈ correct_vals
        correct_vals = [Vec((0.2f0,)),Vec((0.2f0,)),Vec((0.2f0,)),Vec((0.2f0,))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,2.f0)
        @test Vector(Vals) ≈ correct_vals
        correct_vals = [Vec((0.3f0,)),Vec((0.3f0,)),Vec((0.3f0,)),Vec((0.3f0,))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,2.1f0)
        @test Vector(Vals) ≈ correct_vals
    end


    @testset "ConductivityToDiffusivityCoefficient" begin
        eigvec = Vec((1.f0,0.f0))
        eigval = -1.f0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval,0.f0))),
        )
        ctdc = ConductivityToDiffusivityCoefficient(
            stc,
            ConstantCoefficient(2.f0),
            ConstantCoefficient(0.5f0),
        )
        coeff_cache = setup_coefficient_cache(ctdc, qr, sdh)
        correct_vals = [Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0)),Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0)),Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0)),Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,0.f0)
        @test Vector(Vals) ≈ correct_vals
        correct_vals = [Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0)),Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0)),Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0)),Tensor{2,2,Float32}((-1.0,0.0,0.0,0.0))]
        Vals = correct_vals |> similar  |> cu
        @cuda blocks=1 threads=n_cells coeffs_kernel!(Vals,dh,coeff_cache,cellvalues,1.f0)
        @test Vector(Vals) ≈ correct_vals
        
    end

end