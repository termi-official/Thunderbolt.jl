include("contractile_cuboid.jl")

using LsqFit
using GLMakie
using DelimitedFiles

# Helper to plot equibiaxial model
function plot_equibiaxial(axis, P; λₘᵢₙ = 1.08, λₘₐₓ = 1.3, num_samples=25)
    λ₀ = 0.0 #λₘᵢₙ-1.0 # correction in LinYin?
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

f₀ = Tensors.Vec((1.0, 0.0, 0.0))
s₀ = Tensors.Vec((0.0, 1.0, 0.0))
n₀ = Tensors.Vec((0.0, 0.0, 1.0))


f = Figure()
ax = Axis(f[1,1])

plot_equibiaxial(ax, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_passive)[1])
linyin1998_4_fiber_passive = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Fiber_Passive.csv", ',')
scatter!(ax, linyin1998_4_fiber_passive)
linyin1998_4_sheet_passive = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Sheet_Passive.csv", ',')
scatter!(ax, linyin1998_4_sheet_passive)

plot_equibiaxial(ax, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_active)[1])
linyin1998_4_fiber_active = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Fiber_Active.csv", ',')
scatter!(ax, linyin1998_4_fiber_active)
linyin1998_4_sheet_active = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Sheet_Active.csv", ',')
scatter!(ax, linyin1998_4_sheet_active)

#TODO (Fig 12, from Lin&Yin 1998)
# 1. refit parameters of the proposed energies (passive with Levenberg-Marquard and active with Luigi's approach)
function equibiaxial_evaluation_passive(λset, p, model_construction::Function)
    mp = model_construction(p)
    Pset = zero(λset)
    for i ∈ 1:size(λset,1)
        λ = λset[i,:]
        # Assume incompressibility to construct deformation gradient
        F1 = Tensor{2,3,Float64}((
            λ[1],0.0,0.0,
            0.0,λ[1],0.0,
            0.0,0.0,1.0/λ[1]^2))
        P1 = constitutive_driver(F1, f₀, s₀, n₀, 0.0, mp)[1]
        Pset[i,1] = P1[1,1]

        # Assume incompressibility to construct deformation gradient
        F2 = Tensor{2,3,Float64}((
            λ[2],0.0,0.0,
            0.0,λ[2],0.0,
            0.0,0.0,1.0/λ[2]^2))
        P2 = constitutive_driver(F2, f₀, s₀, n₀, 0.0, mp)[1]
        Pset[i,2] = P2[2,2]
    end
    # not sure why we have to flatten the matrix here...
    return vec(Pset)
end

# preprocessing of experimental data 
# 1. shift stretchs to equilibrium via passive measurement data
λshift = minimum(linyin1998_4_fiber_passive[:,1])-1.0
linyin1998_4_fiber_passive[:,1] .-= λshift
linyin1998_4_sheet_passive[:,1] .-= λshift
linyin1998_4_fiber_active[:,1] .-= λshift
linyin1998_4_sheet_active[:,1] .-= λshift

# 2. kPa to original unit
# linyin1998_4_fiber_passive[:,2] .*= 10.0 #(scale)
# linyin1998_4_sheet_passive[:,2] .*= 10.0 #(scale)
# linyin1998_4_fiber_active[:,2] .*= 10.0 #(scale)
# linyin1998_4_sheet_active[:,2] .*= 10.0 #(scale)

model = (λ,p_)->equibiaxial_evaluation_passive(λ, p_, p->GeneralizedHillModel(
    LinYinPassiveModel(p[1],p[2],p[3],p[4],NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model())
)
xdata = hcat(linyin1998_4_fiber_passive[:,1], linyin1998_4_sheet_passive[:,1])
# not sure why we have to flatten the matrix here...
ydata = vec(hcat(linyin1998_4_fiber_passive[:,2], linyin1998_4_sheet_passive[:,2]))
fit = curve_fit(model,
    xdata,
    ydata,
    # Take original parameters as initial guess
    [linyin_original_passive.passive_spring.C₁, linyin_original_passive.passive_spring.C₂, linyin_original_passive.passive_spring.C₃, linyin_original_passive.passive_spring.C₄]
)

p = fit.param
linyin_refit_passive = GeneralizedHillModel(
    LinYinPassiveModel(p[1], p[2], p[3], p[4], NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()
)

f2 = Figure()
ax2 = Axis(f2[1,1])

plot_equibiaxial(ax2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_refit_passive)[1]; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))

# ------

model = (λ,p_)->equibiaxial_evaluation_passive(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p[1],p[2],p[3],p[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model())
)
fit = curve_fit(model,
    xdata,
    ydata,
    [0.1, 0.1, 0.1, 0.1]
)

p_ho_refit_pass = fit.param
ho2009a_refit_passive = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()
)
plot_equibiaxial(ax2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009a_refit_passive)[1]; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))

# -----
# add raw data
scatter!(ax2, linyin1998_4_fiber_passive)
scatter!(ax2, linyin1998_4_sheet_passive)

# 2. refit total stress with different active energies for the generalized Hill model
function equibiaxial_evaluation_active(λset, p, model_construction::Function)
    mp = model_construction(p)
    Pset = zero(λset)
    for i ∈ 1:35
        λ = λset[i]
        # Assume incompressibility to construct deformation gradient
        F1 = Tensor{2,3,Float64}((
            λ,0.0,0.0,
            0.0,λ,0.0,
            0.0,0.0,1.0/λ^2))
        P1 = constitutive_driver(F1, f₀, s₀, n₀, 1.0, mp)[1]
        Pset[i] = P1[1,1]
    end
    for i ∈ 36:length(λset)
        λ = λset[i]
        # Assume incompressibility to construct deformation gradient
        F2 = Tensor{2,3,Float64}((
            λ,0.0,0.0,
            0.0,λ,0.0,
            0.0,0.0,1.0/λ^2))
        P2 = constitutive_driver(F2, f₀, s₀, n₀, 1.0, mp)[1]
        Pset[i] = P2[2,2]
    end
    return Pset
end

new_xdata = [linyin1998_4_fiber_active[:,1]; linyin1998_4_sheet_active[:,1]]
# not sure why we have to flatten the matrix here...
new_ydata = [linyin1998_4_fiber_active[:,2]; linyin1998_4_sheet_active[:,2]]

new_model_hoho = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(HolzapfelOgden2009Model(p[1],p[2],p[3],p[4],0.0,1.0,0.0,1.0,NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model())
)
new_fit = curve_fit(new_model_hoho,
    new_xdata,
    new_ydata,
    [0.1, 0.1, 0.1, 0.1]
)

p_new_model_hoho = new_fit.param
new_model_hoho_fit = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(HolzapfelOgden2009Model(p_new_model_hoho[1],p_new_model_hoho[2],p_new_model_hoho[3],p_new_model_hoho[4],0.0,1.0,0.0,1.0,NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()
)

plot_equibiaxial(ax2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1]; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

# add raw data
scatter!(ax2, linyin1998_4_fiber_active)
scatter!(ax2, linyin1998_4_sheet_active)

