include("contractile_cuboid.jl")

using GLMakie
using DelimitedFiles

# Helper to plot equibiaxial model
function plot_equibiaxial(axis, P; λₘᵢₙ = 1.08, λₘₐₓ = 1.3, num_samples=25)
    λ₀ = λₘᵢₙ-1.0 # correction in LinYin?
    λset = range(λₘᵢₙ, λₘₐₓ, length=num_samples)
    lines!(axis, λset, λ -> P(Tensor{2,3,Float64}((
                                (λ-λ₀),0.0,0.0,
                                 0.0,(λ-λ₀),0.0,
                                 0.0,0.0,1.0/(λ-λ₀)^2))
                            )[1,1]
    )
    lines!(axis, λset, λ -> P(Tensor{2,3,Float64}((
                                (λ-λ₀),0.0,0.0,
                                 0.0,(λ-λ₀),0.0,
                                 0.0,0.0,1.0/(λ-λ₀)^2)
                                )
                            )[2,2]
    )
end

linyin_original_passive = GeneralizedHillModel(
    LinYinPassiveModel(;mpU=NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()
)

linyin_original_active = GeneralizedHillModel(
    LinYinActiveModel(;mpU=NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()
)

f₀ = Vec((1.0, 0.0, 0.0))
s₀ = Vec((0.0, 1.0, 0.0))
n₀ = Vec((0.0, 0.0, 1.0))


f = Figure()
ax = Axis(f[1,1])

plot_equibiaxial(ax, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_passive)[1])
linyin1998_4_fiber_passive = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Fiber_Passive.csv", ',')
linyin1998_4_fiber_passive[:,2] .*= 10.0 #(scale)
scatter!(ax, linyin1998_4_fiber_passive)
linyin1998_4_sheet_passive = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Sheet_Passive.csv", ',')
linyin1998_4_sheet_passive[:,2] .*= 10.0 #(scale)
scatter!(ax, linyin1998_4_sheet_passive)

plot_equibiaxial(ax, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_active)[1])
linyin1998_4_fiber_active = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Fiber_Active.csv", ',')
linyin1998_4_fiber_active[:,2] .*= 10.0 #(scale)
scatter!(ax, linyin1998_4_fiber_active)
linyin1998_4_sheet_active = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Sheet_Active.csv", ',')
linyin1998_4_sheet_active[:,2] .*= 10.0 #(scale)
scatter!(ax, linyin1998_4_sheet_active)

f

#TODO (Fig 12, from Lin&Yin 1998)
# 1. refit parameters of the proposed energies (passive with Levenberg-Marquard and active with Luigi's approach)
function equibiaxial_evaluation_passive(λ, p, model_construction::Function)
    mp = model_construction(p)
    # Assume incompressibility to construct deformation gradient
    F1 = Tensor{2,3,Float64}((
        λ[1],0.0,0.0,
        0.0,λ[1],0.0,
        0.0,0.0,1.0/λ[1]^2))
    P1 = constitutive_driver(F1, f₀, s₀, n₀, 0.0, mp)[1]

    # Assume incompressibility to construct deformation gradient
    F2 = Tensor{2,3,Float64}((
        λ[2],0.0,0.0,
        0.0,λ[2],0.0,
        0.0,0.0,1.0/λ[2]^2))
    P2 = constitutive_driver(F2, f₀, s₀, n₀, 0.0, mp)[1]

    P1[1,1], P2[2,2]
end

# TODO preprocessing of experimental data - shift stretchs to equilibrium via passive measurement data
fit = curve_fit((λ,p_)->equibiaxial_evaluation_passive(λ, p_, p->GeneralizedHillModel(
    LinYinPassiveModel(p[1],p[2],p[3],p[4],NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()),
    linyin1998_4_fiber_passive[:,1],
    linyin1998_4_fiber_passive[:,2],

))

# 2. refit total stress with different active energies for the generalized Hill model
# 3. refilt total stress with the same active energy, but holzapfel-Ogden 2009 passive energy (anisotropic form)
