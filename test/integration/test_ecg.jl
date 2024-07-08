@testset "ECG" begin
    @testset "Blocks with $geo" for geo in (Hexahedron, Tetrahedron)
        size = 4.
        signal_strength = 0.04
        nel_heart = (18, 19, 17)
        nel_torso = (17, 15, 19)
        ground_vertex = Vec(-size, -size, -size)
        electrodes = [
            ground_vertex,
            Vec(-size,  0.,  0.), 
            Vec( size,  0.,  0.),
            Vec( 0., -size,  0.),
            Vec( 0.,  size,  0.),
            Vec( 0.,  0., -size),
            Vec( 0.,  0.,  size),
        ]
        electrode_pairs = [(i,1) for i in 2:length(electrodes)]

        heart_grid = generate_mesh(geo, nel_heart)
        Ferrite.transform_coordinates!(heart_grid, x->Vec{3}(sign.(x) .* x.^2))

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

        torso_grid = generate_mesh(geo, nel_torso, Vec((-size,-size,-size)), Vec((size,size,size)))

        u = zeros(Thunderbolt.solution_size(heart_fun))
        plonsey_ecg = Thunderbolt.Plonsey1964ECGGaussCache(op, u)
        poisson_ecg = Thunderbolt.PoissonECGReconstructionCache(heart_fun, torso_grid, ConstantCoefficient(κᵢ), ConstantCoefficient(κ), electrodes)

        @testset "Equilibrium" begin
            u .= 0.0
            @testset "Plonsey1964" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                for electrode in electrodes
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, electrode, 1.0) ≈ 0.0
                end
            end

            @testset "Poisson" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg)
                @test length(ecg_vals) == length(electrodes)
                @test all(ecg_vals .≈ 0.0)
            end
        end

        @testset "Planar wave dim=$dim" for dim in 1:3
            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->x[dim]^3)

            @testset "Plonsey1964 xᵢ³" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? size : 0.0 for i in 1:3]),1.0) > signal_strength
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? -size : 0.0 for i in 1:3]),1.0) < signal_strength
                for dim2 in 1:3
                    dim2 == dim && continue
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? size : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? -size : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                end
            end

            @testset "Poisson xᵢ³" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg)
                @test ecg_vals[1] ≈ 0.0 atol=1e-12 # Ground
                for dim2 in 1:3
                    if dim2 == dim
                        @test ecg_vals[2*dim2+1]-ecg_vals[2*dim2] > 0.0
                    else
                        @test ecg_vals[2*dim2+1]-ecg_vals[2*dim2] ≈ 0.0 atol=1e-4
                    end
                end
            end

            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->-x[dim]^3)

            @testset "Plonsey1964 -xᵢ³" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? size : 0.0 for i in 1:3]),1.0) < signal_strength
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim ? -size : 0.0 for i in 1:3]),1.0) > signal_strength
                for dim2 in 1:3
                    dim2 == dim && continue
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? size : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                    @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec{3}([i==dim2 ? -size : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                end
            end

            @testset "Poisson -xᵢ³" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg) 
                @test ecg_vals[1] ≈ 0.0 atol=1e-12 # Ground
                for i in 1:3
                    ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg) 
                    
                    if i == dim
                        @test ecg_vals[2*i+1]-ecg_vals[2*i] < 0.0
                    else
                        @test ecg_vals[2*i+1]-ecg_vals[2*i] ≈ 0.0 atol=1e-4
                    end
                end
            end
        end

        @testset "Symmetric stimuli" begin
            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->sqrt(3)-norm(x))

            @testset "Plonsey1964 √3-||x||" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( size,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-size, 0.0, 0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-size,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0, size, 0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0, size,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0,-size, 0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-size,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0, 0.0, size),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0, size),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec( 0.0, 0.0,-size),1.0) atol=1e-2
            end

            @testset "Poisson √3-||x||" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg)
                @test ecg_vals[1] ≈ 0.0 atol=1e-12 # Ground
                for i in 3:length(ecg_vals)
                    @test ecg_vals[2] ≈ ecg_vals[i] atol=1e-1
                end
            end

            Ferrite.apply_analytical!(u, heart_fun.dh, :φₘ, x->x[1]^2)

            @testset "Plonsey1964 x₁²" begin
                Thunderbolt.update_ecg!(plonsey_ecg, u)
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(size,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(-size,0.0,0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,size,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-size,0.0),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,-size,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,size),1.0) atol=1e-2
                @test Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,size),1.0) ≈ Thunderbolt.evaluate_ecg(plonsey_ecg, Vec(0.0,0.0,-size),1.0) atol=1e-2
            end

            @testset "Poisson x₁²" begin
                Thunderbolt.update_ecg!(poisson_ecg, u)
                ecg_vals = Thunderbolt.evaluate_ecg(poisson_ecg)
                @test ecg_vals[1] ≈ 0.0 atol=1e-12 # Ground
                for dim in 1:3
                    @test ecg_vals[2dim+1] ≈ ecg_vals[2dim] atol=1e-1
                end
            end
        end
    end
end
