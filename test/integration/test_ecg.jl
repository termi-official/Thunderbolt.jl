@testset "PseudoECG" begin
    grid = generate_grid(geo, (10, 8, 12))
    Ferrite.transform_coordinates!(grid, x->Vec{3}(x.^3))
    cs = CartesianCoordinateSystem(grid)

    qr_collection = QuadratureRuleCollection(2)
    ip_collection = LagrangeCollection{1}()

    ms = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec(0., 0., 1.)),
        ConstantCoefficient(Vec(1., 0., 0.)),
        ConstantCoefficient(Vec(0., 1., 0.))
    )

    κ = ConstantCoefficient(SymmetricTensor{2,3}((
        1, 0, 0,
            1, 0,
                1
    )))
    # xref copy pasta from examples/leads

    model = MonodomainModel(
        ConstantCoefficient(1),
        ConstantCoefficient(1),
        κ,
        Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x,t) -> norm(x) < 1.5 && t < 2.0 ? 0.5 : 0.0, Thunderbolt.CoordinateSystemCoefficient(cs)),
            [SVector((0.0, 2.1))]
        ),
        Thunderbolt.PCG2019()
    )

    problem = semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => ip_collection)),
        grid
    )
    @testset "Plonsey1964 3D $geo" for (refshape, geo) in ((RefHexahedron, Hexahedron), (RefTetrahedron, Tetrahedron))
        lead_field = Thunderbolt.Geselowitz1989ECGLeadCache(problem, κ, κ, [Vec(1., 0., 0.), Vec(-1., 0., 0.)], [(1,2)])

        ecg_reconst_cache = Thunderbolt.Potse2006ECGPoissonReconstructionCache(problem, κ, κ)

        u = zeros(Thunderbolt.solution_size(problem.A))

        κ = ConstantCoefficient(SymmetricTensor{2,3,Float64}((1.0, 0, 0, 1.0, 0, 1.0)))
        
        @testset "Equilibrium" begin
            u .= 0.0
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec(2.0,0.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,2.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,-2.0),1.0) ≈ 0.0
        end

        @testset "Planar wave dim=$dim" for dim in 1:3
            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->x[dim]^3)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) > 0.1
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) < 0.1
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
            end

            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->-x[dim]^3)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) < 0.1
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) > 0.1
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
            end
        end

        @testset "Symmetric stimuli" begin
            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->sqrt(3)-norm(x))
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec( 2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(-2.0, 0.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 2.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0, 2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0,-2.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 0.0, 2.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0, 2.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 0.0,-2.0),1.0) atol=1e-2

            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->x[1]^2)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec(2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,-2.0),1.0) atol=1e-2
        end
    end

    @testset "Plonsey1964 3D $geo" for (refshape, geo) in ((RefHexahedron, Hexahedron), (RefTetrahedron, Tetrahedron))
        lead_field = Thunderbolt.Geselowitz1989ECGLeadCache(problem, κ, κ, [Vec(1., 0., 0.), Vec(-1., 0., 0.)], [(1,2)])

        ecg_reconst_cache = Thunderbolt.Potse2006ECGPoissonReconstructionCache(problem, κ, κ)

        u = zeros(Thunderbolt.solution_size(problem.A))

        κ = ConstantCoefficient(SymmetricTensor{2,3,Float64}((1.0, 0, 0, 1.0, 0, 1.0)))
        
        @testset "Equilibrium" begin
            u .= 0.0
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec(2.0,0.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,2.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,-2.0),1.0) ≈ 0.0
        end

        @testset "Planar wave dim=$dim" for dim in 1:3
            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->x[dim]^3)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) > 0.1
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) < 0.1
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
            end

            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->-x[dim]^3)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) < 0.1
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) > 0.1
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
            end
        end

        @testset "Symmetric stimuli" begin
            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->sqrt(3)-norm(x))
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec( 2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(-2.0, 0.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 2.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0, 2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0,-2.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 0.0, 2.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0, 2.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 0.0,-2.0),1.0) atol=1e-2

            Ferrite.apply_analytical!(u, problem.A.dh, :ϕₘ, x->x[1]^2)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(problem, κ)
            reinit!(pecg, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec(2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,-2.0),1.0) atol=1e-2
        end
    end
end
