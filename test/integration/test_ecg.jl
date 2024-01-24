@testset "PseudoECG" begin
    @testset "Plonsey1964 3D $geo" for (refshape, geo) in ((RefHexahedron, Hexahedron), (RefTetrahedron, Tetrahedron))
        grid = generate_grid(geo, (10, 8, 12))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(x.^3))
        ip = Lagrange{refshape,1}()

        dh = DofHandler(grid)
        Ferrite.add!(dh, :ϕₘ, ip)
        close!(dh);
        u = zeros(ndofs(dh))

        # xref copy pasta from src/solver/euler.jl#setup_solver_caches
        qr = QuadratureRule{refshape}(2)
        cv = CellValues(qr, ip)
        op = Thunderbolt.AssembledBilinearOperator(
            create_sparsity_pattern(dh),
            Thunderbolt.BilinearDiffusionElementCache(
                Thunderbolt.BilinearDiffusionIntegrator(
                    ConstantCoefficient(SymmetricTensor{2,3,Float64}((1.0, 0, 0, 1.0, 0, 1.0))),
                ),
                cv,
            ),
            dh,
        )
        Thunderbolt.update_operator!(op, 0.0) # trigger assembly

        @testset "Equilibrium" begin
            u .= 0.0
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(dh, op, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec(2.0,0.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,2.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) ≈ 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,-2.0),1.0) ≈ 0.0
        end

        @testset "Planar wave dim=$dim" for dim in 1:3
            Ferrite.apply_analytical!(u, dh, :ϕₘ, x->x[dim]^3)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(dh, op, u)

            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) > 0.1
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) < 0.1
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
            end

            Ferrite.apply_analytical!(u, dh, :ϕₘ, x->-x[dim]^3)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(dh, op, u)

            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) < 0.1
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) > 0.1
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0 atol=1e-4
            end
        end

        @testset "Symmetric stimuli" begin
            Ferrite.apply_analytical!(u, dh, :ϕₘ, x->sqrt(3)-norm(x))
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(dh, op, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec( 2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(-2.0, 0.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 2.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0, 2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0,-2.0, 0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 0.0, 2.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0, 2.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec( 0.0, 0.0,-2.0),1.0) atol=1e-2

            Ferrite.apply_analytical!(u, dh, :ϕₘ, x->x[1]^2)
            pecg = Thunderbolt.Plonsey1964ECGGaussCache(dh, op, u)
            @test Thunderbolt.evaluate_ecg(pecg, Vec(2.0,0.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(-2.0,0.0,0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,-2.0,0.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) atol=1e-2
            @test Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,2.0),1.0) ≈ Thunderbolt.evaluate_ecg(pecg, Vec(0.0,0.0,-2.0),1.0) atol=1e-2
        end
    end
end
