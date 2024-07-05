@testset "ECG"
    @testset "Blocks with $geo" for geo in (Hexahedron, Tetrahedron)
        nel_heart = (5, 4, 6)
        nel_torso = (4, 2, 5)
        ground_vertex = Vec(-1., -1., -1.)
        electrodes = [
            ground_vertex,
            Vec(0., 0., -1.), 
            Vec(0., 0., 1.),
            Vec(0., 1., 0.),
            Vec(0., -1., 0.),
            Vec(1., 0., 0.),
            Vec(-1., 0., 0.),
            ]
        electrode_pairs = [(i,1) for i in 2:length(electrodes)]

        heart_grid = generate_mesh(geo, nel_heart)
        Ferrite.transform_coordinates!(heart_grid, x->Vec{3}(x.^3))

        κ  = SymmetricTensor{2,3,Float64}((1.0, 0, 0, 1.0, 0, 1.0))
        κᵢ = SymmetricTensor{2,3,Float64}((1.0, 0, 0, 1.0, 0, 1.0))

        heart_model = TransientDiffusionModel(
            ConstantCoefficient(κ),
            NoStimulationProtocol(), # Poisoning to detecte if we accidentally touch these
            :φₘ
        )
        heart_fun = semidiscretize(
            heart_model,
            FiniteElementDiscretization(
                Dict(:φₘ => LagrangeCollection{1}()),
                Dirichlet[]
            ),
            heart_grid
        )

        op = Thunderbolt.setup_assembled_operator(
            Thunderbolt.BilinearDiffusionIntegrator(
                ConstantCoefficient(κ),
            ),
            Thunderbolt.SparseMatrixCSC,
            heart_fun.dh, :φₘ, QuadratureRuleCollection(1), 
        )
        Thunderbolt.update_operator!(op, 0.0) # trigger assembly

        torso_grid = generate_mesh(geo, nel_torso, Vec((-2.0,-2.0,-2.0)), Vec((2.0,2.0,2.0)))

        u = zeros(Thunderbolt.solution_size(heart_fun))
        plonsey_ecg = Thunderbolt.Plonsey1964ECGGaussCache(op, u)
        poisson_ecg = Thunderbolt.Potse2006ECGPoissonReconstructionCache(heart_fun, torso_grid, ConstantCoefficient(κᵢ), ConstantCoefficient(κ), electrodes)

        @testset "Equilibrium" begin
            u .= 0.0
            @testset "Plonsey1964" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(2.0,0.0,0.0),1.0) ≈ 0.0
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-2.0,0.0,0.0),1.0) ≈ 0.0
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,2.0,0.0),1.0) ≈ 0.0
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-2.0,0.0),1.0) ≈ 0.0
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,2.0),1.0) ≈ 0.0
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,-2.0),1.0) ≈ 0.0
            end

            @testset "Poisson" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                for i in 1:length(electrode_pairs)
                    @test Thunderbolt.evaluate_ecg(poisson_ecg, electrodes[i]) ≈ 0.0
                end
            end
        end

        @testset "Planar wave dim=$dim" for dim in 1:3
            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->x[dim]^3)

            @testset "Plonsey1964" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) > 0.1
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) < 0.1
                for dim2 in 1:3
                    dim2 == dim && continue
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                end
            end

            @testset "Poisson" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg) 
                for i in 1:length(electrode_pairs)
                    @test ecg_vals[i] < 0.1
                end
            end

            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->-x[dim]^3)

            @testset "Plonsey1964" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) < 0.1
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) > 0.1
                for dim2 in 1:3
                    dim2 == dim && continue
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                end
            end
        end

        @testset "Symmetric stimuli" begin
            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->sqrt(3)-norm(x))

            @testset "Plonsey1964" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-2.0, 0.0, 0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0, 2.0, 0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0, 2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0,-2.0, 0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0, 0.0, 2.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0, 2.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0, 0.0,-2.0),1.0) atol=1e-2
            end


            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->x[1]^2)

            @testset "Plonsey1964" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-2.0,0.0,0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-2.0,0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,2.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,2.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,-2.0),1.0) atol=1e-2
            end
        end
end
