@testset "Type Stability" begin
    f₀ = Tensors.Vec{3,Float64}((1.0,0.0,0.0))
    s₀ = Tensors.Vec{3,Float64}((0.0,1.0,0.0))
    n₀ = Tensors.Vec{3,Float64}((0.0,0.0,1.0))
    F = one(Tensors.Tensor{2,3})
    Caᵢ = 1.0

    material_model_set = [
        NullEnergyModel(),
        HolzapfelOgden2009Model(),
        HolzapfelOgden2009Model(;mpU = SimpleCompressionPenalty()),
        HolzapfelOgden2009Model(;mpU = NeffCompressionPenalty()),
        TransverseIsotopicNeoHookeanModel(),
        LinYinPassiveModel(),
        LinYinActiveModel(),
        HumphreyStrumpfYinModel(),
    ]
    @testset "Energies $material_model" for material_model ∈ material_model_set
        @test_call Thunderbolt.Ψ(F, f₀, s₀, n₀, material_model)
    end

    @testset "Constitutive Models" begin
        active_stress_set = [
            SimpleActiveStress(),
            PiersantiActiveStress(),
        ]
        @testset failfast=true "Active Stress" for active_stress ∈ active_stress_set
            @testset let passive_spring = material_model_set[rand(1:length(material_model_set))]
                model = ActiveStressModel(
                    passive_spring,
                    PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
                    PelceSunLangeveld1995Model()
                )
                @test_call constitutive_driver(F, f₀, s₀, n₀, Caᵢ, model)
            end
        end

        contraction_model_set = [
            ConstantStretchModel(),
            PelceSunLangeveld1995Model(),
        ]
        Fᵃmodel_set = [
            GMKActiveDeformationGradientModel(),
            GMKIncompressibleActiveDeformationGradientModel(),
            RLRSQActiveDeformationGradientModel(0.75),
        ]
        @testset failfast=true "Extended Hill" begin
            @testset "Active Deformation Gradient" for Fᵃmodel ∈ Fᵃmodel_set
                @testset "Contraction Model" for contraction_model ∈ contraction_model_set
                    @testset let passive_spring = material_model_set[rand(1:length(material_model_set))]
                        model = ExtendedHillModel(
                            passive_spring,
                            ActiveMaterialAdapter(passive_spring),
                            Fᵃmodel,
                            contraction_model,
                        )
                        @test_call broken=true constitutive_driver(F, f₀, s₀, n₀, Caᵢ, model)
                    end
                end
            end
        end
    end
end