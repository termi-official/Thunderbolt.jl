@testset "PseudoECG" begin
    @testset "Plonsey1964" begin
        grid = generate_grid(Hexahedron, (1, 1, 1))
        ip = Lagrange{RefHexahedron,1}()

        dh = DofHandler(grid)
        Ferrite.add!(dh, :ϕₘ, ip)
        close!(dh);
        u = zeros(ndofs(dh))

        # xref copy pasta from src/solver/euler.jl#setup_solver_caches
        qr = QuadratureRule{RefHexahedron}(2)
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

            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? 2.0 : 0.0 for i in 1:3]),1.0) > 0.0
            @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim ? -2.0 : 0.0 for i in 1:3]),1.0) < 0.0
            for dim2 in 1:3
                dim2 == dim && continue
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? 2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0
                @test Thunderbolt.evaluate_ecg(pecg, Vec{3}([i==dim2 ? -2.0 : 0.0 for i in 1:3]),1.0) ≈ 0.0
            end
        end
    end
end
