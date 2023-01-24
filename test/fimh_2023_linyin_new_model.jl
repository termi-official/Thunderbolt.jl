include("contractile_cuboid.jl")

using LsqFit
using DelimitedFiles

set_theme!(theme_ggplot2())

# Helper to plot equibiaxial model
function plot_equibiaxial(axis, comp1, comp2, P, label_base, color; λ₀ = 0.0, λₘᵢₙ = 1.08, λₘₐₓ = 1.3, num_samples=25)
    λset = range(λₘᵢₙ, λₘₐₓ, length=num_samples)
    lines!(axis, λset, λ -> P(Tensor{2,3,Float64}((
                                (λ-λ₀),0.0,0.0,
                                0.0,(λ-λ₀),0.0,
                                0.0,0.0,1.0/(λ-λ₀)^2))
                            )[comp1,comp2];
    label="$label_base",
    linewidth = 3,
    linestyle = nothing,
    color = color,
    #marker = marker,
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

linyin1998_4_fiber_passive = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Fiber_Passive.csv", ',')
linyin1998_4_sheet_passive = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Sheet_Passive.csv", ',')
linyin1998_4_fiber_active = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Fiber_Active.csv", ',')
linyin1998_4_sheet_active = readdlm("./data/experiments/LinYin1998/LinYin_1998_Fig4_Sheet_Active.csv", ',')

# f = Figure()
# ax = Axis(f[1,1]; xlabel="λ", ylabel="stress (kPa)", subtitle="Equibiaxial (passive)")
# plot_equibiaxial(ax, 1, 1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_passive)[1], "Lin Yin (1998) passive", :red; λ₀ = 0.08)
# plot_equibiaxial(ax, 2, 2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_passive)[1], "", :red; λ₀ = 0.08)
# plot_equibiaxial(ax, 3, 3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_passive)[1], "", :red; λ₀ = 0.08)
# scatter!(ax, linyin1998_4_fiber_passive, label="Lin Yin (1998) experiment 4 (fiber, passive)")
# scatter!(ax, linyin1998_4_sheet_passive, label="Lin Yin (1998) experiment 4 (sheet, passive)")
# plot_equibiaxial(ax, 1, 1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_active)[1], "Lin Yin (1998) active", :blue; λ₀ = 0.08)
# plot_equibiaxial(ax, 2, 2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_active)[1], "", :blue; λ₀ = 0.08)
# plot_equibiaxial(ax, 3, 3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_original_active)[1], "", :blue; λ₀ = 0.08)
# scatter!(ax, linyin1998_4_fiber_active, label="Lin Yin (1998) experiment 4 (fiber, active)")
# scatter!(ax, linyin1998_4_sheet_active, label="Lin Yin (1998) experiment 4 (sheet, active)")

# axislegend(ax; position=:lt)

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

xdata = hcat(linyin1998_4_fiber_passive[:,1], linyin1998_4_sheet_passive[:,1])
# not sure why we have to flatten the matrix here...
ydata = vec(hcat(linyin1998_4_fiber_passive[:,2], linyin1998_4_sheet_passive[:,2]))

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
fit = curve_fit(model,
    xdata,
    ydata,
    # Take original parameters as initial guess
    [linyin_original_passive.passive_spring.C₁, linyin_original_passive.passive_spring.C₂, linyin_original_passive.passive_spring.C₃, linyin_original_passive.passive_spring.C₄]
)

linyin_pass_refit_p = fit.param
linyin_refit_passive = GeneralizedHillModel(
    LinYinPassiveModel(linyin_pass_refit_p[1], linyin_pass_refit_p[2], linyin_pass_refit_p[3], linyin_pass_refit_p[4], NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(0.5),
    PelceSunLangeveld1995Model()
)

sheet_portion_rlrsq = 0.75

size_inches = (6, 2)
size_pt = 200 .* size_inches

f2 = Figure(resolution = size_pt, fontsize = 20)
ax2p = Axis(f2[1,1]; xlabel="λ", ylabel="stress (kPa)", subtitle="Equibiaxial (passive)")
ax2a = Axis(f2[1,2]; xlabel="λ", ylabel="stress (kPa)", subtitle="Equibiaxial (active)")

plot_equibiaxial(ax2p, 1, 1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_refit_passive)[1], "Lin-Yin (1998)", :red; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))
plot_equibiaxial(ax2p, 2, 2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_refit_passive)[1], "", :red; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))
plot_equibiaxial(ax2p, 3, 3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_refit_passive)[1], "", :red; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))

# ------

model = (λ,p_)->equibiaxial_evaluation_passive(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p[1],p[2],p[3],p[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NullEnergyModel()),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
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
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)
plot_equibiaxial(ax2p, 1, 1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009a_refit_passive)[1], "Holzapfel-Ogden (2008)", :blue; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))
plot_equibiaxial(ax2p, 2, 2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009a_refit_passive)[1], "", :blue; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))
plot_equibiaxial(ax2p, 3, 3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009a_refit_passive)[1], "", :blue; λₘᵢₙ=minimum(xdata), λₘₐₓ=maximum(xdata))

# -----
# add raw data
scatter!(ax2p, linyin1998_4_fiber_passive, label="Lin Yin (1998) experiment 4 (fiber)", marker=:xcross, markersize=20, color=:black)
scatter!(ax2p, linyin1998_4_sheet_passive, label="Lin Yin (1998) experiment 4 (sheet)", marker=:cross, markersize=20, color=:black)

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
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model())
)
new_fit = curve_fit(new_model_hoho,
    new_xdata,
    new_ydata,
    [0.1, 0.1, 0.1, 0.1]
)
@assert new_fit.converged

p_new_model_hoho = new_fit.param
new_model_hoho_fit = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(HolzapfelOgden2009Model(p_new_model_hoho[1],p_new_model_hoho[2],p_new_model_hoho[3],p_new_model_hoho[4],0.0,1.0,0.0,1.0,NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)

plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1], "Holzapfel-Ogden (2008)", :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1], "", :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1], "", :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

# New model
Base.@kwdef struct NewActiveSpring
	a   = 1.0
	aᶠ  = 1.0
	mpU = NullCompressionModel()
end

function Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring)
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
	I₃ = det(C)
	J = det(F)
    I₁ = tr(C/cbrt(J^2))
	I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0)^2 + U(I₃, mpU)
end

Base.@kwdef struct NewActiveSpring2
	a   = 1.0
	aᶠ  = 1.0
	mpU = NullCompressionModel()
end

function Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring2)
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
	I₃ = det(C)
	J = det(F)
    I₁ = tr(C/cbrt(J^2))
	I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0) + U(I₃, mpU)
end

Base.@kwdef struct NewActiveSpring3
	a   = 1.0
	aᶠ  = 1.0
	mpU = NullCompressionModel()
end

function Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring3)
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
	I₃ = det(C)
	J = det(F)
    I₁ = tr(C/cbrt(J^2))
	I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0) + aᶠ/2.0*(I₄ᶠ-1.0)^2 + U(I₃, mpU)
end


new_model_hopo = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring(p[1],p[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model())
)
new_fit = curve_fit(new_model_hopo,
    new_xdata,
    new_ydata,
    [0.1, 0.1]
)

p_new_model_hopo = new_fit.param
new_model_hopo_fit_22 = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring(p_new_model_hopo[1],p_new_model_hopo[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)
plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_22)[1], "a(Iᵉ₁-3)² + aᶠ(Iᵉ₄ᶠ-1)²", :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_22)[1], "", :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_22)[1], "", :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))



new_model_hopop = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring2(p[1],p[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(1.0),
    ConstantStretchModel())
)
new_fit = curve_fit(new_model_hopop,
    new_xdata,
    new_ydata,
    [0.1, 0.1]
)

p_new_model_hopop = new_fit.param
new_model_hopop_fit_21 = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring2(p_new_model_hopop[1],p_new_model_hopop[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(1.0),
    ConstantStretchModel()
)
plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopop_fit_21)[1], "a(I₁-3)² + aᶠ(I₄ᶠ-1)", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopop_fit_21)[1], "", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopop_fit_21)[1], "", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))



new_model_hopo = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring2(p[1],p[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model())
)
new_fit = curve_fit(new_model_hopo,
    new_xdata,
    new_ydata,
    [0.1, 0.1]
)

p_new_model_hopo = new_fit.param
new_model_hopo_fit_21 = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring2(p_new_model_hopo[1],p_new_model_hopo[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)
plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_21)[1], "a(Iᵉ₁-3)² + aᶠ(Iᵉ₄ᶠ-1)", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_21)[1], "", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_21)[1], "", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))


new_model_hopo = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring3(p[1],p[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model())
)
new_fit = curve_fit(new_model_hopo,
    new_xdata,
    new_ydata,
    [0.1, 0.1]
)

p_new_model_hopo = new_fit.param
new_model_hopo_fit_12 = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(NewActiveSpring3(p_new_model_hopo[1],p_new_model_hopo[2],NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)
plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_12)[1], "a(I₁-3) + aᶠ(I₄ᶠ-1)²", :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_12)[1], "", :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_12)[1], "", :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))


Base.@kwdef struct PiersantiActiveStressEnergy2022
	Tᵃₘₐₓ = 1.0
    nᶠ = 1.0
    nˢ = 1.0
    nⁿ = 1.0
end

function Ψ(F, f₀, s₀, n₀, mp::PiersantiActiveStressEnergy2022)
    @unpack Tᵃₘₐₓ, nᶠ, nˢ, nⁿ = mp

    C = tdot(F)
	I₃ = det(C)
	J = det(F)
	I₄ᶠ = f₀ ⋅ C ⋅ f₀
    I₄ˢ = s₀ ⋅ C ⋅ s₀
    I₄ⁿ = n₀ ⋅ C ⋅ n₀

    return 2Tᵃₘₐₓ *(nᶠ*I₄ᶠ + nˢ*I₄ˢ + nⁿ*I₄ⁿ)
end

new_model_hopa = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(PiersantiActiveStressEnergy2022(1.0,p[1],p[2],p[3])),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model())
)
new_fit = curve_fit(new_model_hopa,
    new_xdata,
    new_ydata,
    [1.0, 0.5, 0.1]
)

p_new_model_hopa = new_fit.param
new_model_hopa_fit = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(PiersantiActiveStressEnergy2022(1.0,p_new_model_hopa[1],p_new_model_hopa[2],p_new_model_hopa[3])),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)
plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopa_fit)[1], "Piersanti et al. (2022)", :orange; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopa_fit)[1], "", :orange; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopa_fit)[1], "", :orange; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))


Base.@kwdef struct ConstantStretchModel 
    λ = 1.0
end
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ

new_model_hop = (λ,p_)->equibiaxial_evaluation_active(λ, p_, p->GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(PiersantiActiveStressEnergy2022(1.0,p[1],p[2],p[3])),
    RLRSQActiveDeformationGradientModel(1.0),
    ConstantStretchModel())
)
new_fit = curve_fit(new_model_hop,
    new_xdata,
    new_ydata,
    [1.0, 0.5, 0.1]
)

p_new_model_hop = new_fit.param
new_model_hop_fit = GeneralizedHillModel(
    HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel()),
    ActiveMaterialAdapter(PiersantiActiveStressEnergy2022(1.0,p_new_model_hop[1],p_new_model_hop[2],p_new_model_hop[3])),
    RLRSQActiveDeformationGradientModel(1.0),
    ConstantStretchModel()
)
plot_equibiaxial(ax2a, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hop_fit)[1], "Piersanti et al. (2022) with λᵃ = 1", :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hop_fit)[1], "Piersanti et al. (2022) with λᵃ = 1", :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax2a, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hop_fit)[1], "Piersanti et al. (2022) with λᵃ = 1", :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))


# add raw data
scatter!(ax2a, linyin1998_4_fiber_active, label="Lin Yin (1998) experiment 4 (fiber)", marker=:xcross, markersize=20, color=:black)
scatter!(ax2a, linyin1998_4_sheet_active, label="Lin Yin (1998) experiment 4 (sheet)", marker=:cross, markersize=20, color=:black)

axislegend(ax2p; position=:lt)
axislegend(ax2a; position=:lt)

# Active strain failure
ho2009_active_strain = GeneralizedHillModel(
    NullEnergyModel(),
    ActiveMaterialAdapter(HolzapfelOgden2009Model(p_ho_refit_pass[1],p_ho_refit_pass[2],p_ho_refit_pass[3],p_ho_refit_pass[4],0.0,1.0,0.0,1.0,NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)
linyin_active_strain = GeneralizedHillModel(
    NullEnergyModel(),
    ActiveMaterialAdapter(LinYinPassiveModel(linyin_pass_refit_p[1], linyin_pass_refit_p[2], linyin_pass_refit_p[3], linyin_pass_refit_p[4], NullCompressionModel())),
    RLRSQActiveDeformationGradientModel(sheet_portion_rlrsq),
    PelceSunLangeveld1995Model()
)


f3 = Figure(resolution = size_pt, fontsize = 24)
ax3f = Axis(f3[1,1]; xlabel="λ", ylabel="stress (kPa)", subtitle="Fiber")
ax3s = Axis(f3[1,2]; xlabel="λ", ylabel="", subtitle="Sheetlet")
ax3n = Axis(f3[1,3]; xlabel="λ", ylabel="", subtitle="Normal")
Label(f3[1, 1:3, Top()], "Equibiaxial Experiment - Active Strain", valign = :bottom,
    font = :bold,
    padding = (0, 5, 32, 0))

plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009_active_strain)[1], "Holzapfel-Ogden (passive)", :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.5, ho2009_active_strain)[1], "Holzapfel-Ogden (half active)", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, ho2009_active_strain)[1], "Holzapfel-Ogden (full active)", :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_active_strain)[1], "Lin-Yin (passive)", :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 0.5, linyin_active_strain)[1], "Lin-Yin (half active)", :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
# plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, linyin_active_strain)[1], "Lin-Yin (active)", :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, linyin_active_strain)[1], "Lin-Yin (full active)", :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=1.1) # Makie overflows with the line above
scatter!(ax3f, linyin1998_4_fiber_passive, label="Lin Yin (1998) fig 4 (passive)", marker=:xcross)
scatter!(ax3f, linyin1998_4_fiber_active, label="Lin Yin (1998) fig 4 (active)", marker=:xcross)
ylims!(ax3f, -5.0, 40.0)

plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009_active_strain)[1], "Holzapfel-Ogden (passive)", :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.5, ho2009_active_strain)[1], "Holzapfel-Ogden (half active)", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, ho2009_active_strain)[1], "Holzapfel-Ogden (full active)", :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_active_strain)[1], "Lin-Yin (passive)", :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 0.5, linyin_active_strain)[1], "Lin-Yin (half active)", :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
# plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, linyin_active_strain)[1], "Lin-Yin (active)", :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, linyin_active_strain)[1], "Lin-Yin (full active)", :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=1.1) # Makie overflows with the line above
scatter!(ax3s, linyin1998_4_sheet_passive, label="Lin Yin (1998) fig 4 (passive)", marker=:xcross)
scatter!(ax3s, linyin1998_4_sheet_active, label="Lin Yin (1998) fig 4 (active)", marker=:xcross)
ylims!(ax3s, -3.0, 35.0)

plot_equibiaxial(ax3n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, ho2009_active_strain)[1], "Holzapfel-Ogden (passive)", :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.5, ho2009_active_strain)[1], "Holzapfel-Ogden (half active)", :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, ho2009_active_strain)[1], "Holzapfel-Ogden (full active)", :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.0, linyin_active_strain)[1], "Lin-Yin (passive)", :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 0.5, linyin_active_strain)[1], "Lin-Yin (half active)", :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax3n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, linyin_active_strain)[1], "Lin-Yin (full active)", :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
scatter!(ax3n, Matrix{Float64}(undef, 0, 2), label="Lin Yin (1998) fig 4 (passive)", marker=:xcross)
scatter!(ax3n, Matrix{Float64}(undef, 0, 2), label="Lin Yin (1998) fig 4 (active)", marker=:xcross)
ylims!(ax3n, -50.0, 50.0)

f3[1,4] = Legend(f3, ax3f, framevisible=false, bgcolor=:white)
# f3[2,:] = Legend(f3, ax3f, framevisible=false)

save("active_strain_failure.eps", f3)


f4 = Figure(resolution = size_pt, fontsize = 20)
ax4f = Axis(f4[1,1]; xlabel="λ", ylabel="stress (kPa)", subtitle="Fiber")
ax4s = Axis(f4[1,2]; xlabel="λ", ylabel="", subtitle="Sheetlet")
ax4n = Axis(f4[1,3]; xlabel="λ", ylabel="", subtitle="Normal")
Label(f4[1, 1:3, Top()], "Equibiaxial Experiment - Active Stress and Our New Model", valign = :bottom,
    font = :bold,
    padding = (0, 5, 32, 0))

label = L"\Psi^{\mathrm{a}} = \sum_d 2 s_d \sqrt{I_{4,d}}"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hop_fit)[1], label, :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hop_fit)[1], label, :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hop_fit)[1], label, :pink; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

label = L"\Psi^{\mathrm{a}} = \sum_d 2 s_d \sqrt{I^{\textrm{e}}_{4,d}}"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopa_fit)[1], label, :orange; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopa_fit)[1], label, :orange; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopa_fit)[1], label, :orange; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

label = L"\Psi^{\mathrm{a}} = a(I^{\textrm{e}}_{1}-3) + \frac{a_f}{2}(I^{\textrm{e}}_{4,f}-1)^2"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_12)[1], label, :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_12)[1], label, :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_12)[1], label, :green; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

label = L"\Psi^{\mathrm{a}} = \frac{a}{2}(I^{\textrm{e}}_{1}-3)^2 + a_f(I^{\textrm{e}}_{4,f}-1)"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_21)[1], label, :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_21)[1], label, :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_21)[1], label, :yellow; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

label = L"\Psi^{\mathrm{a}} = \frac{a}{2}(I_{1}-3)^2 + a_f(I_{4,f}-1)"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopop_fit_21)[1], label, :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopop_fit_21)[1], label, :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopop_fit_21)[1], label, :black; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

label = L"\Psi^{\mathrm{a}} = \frac{a}{2}(I^{\textrm{e}}_{1}-3)^2 + \frac{a_f}{2}(I^{\textrm{e}}_{4,f}-1)^2"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_22)[1], label, :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_22)[1], label, :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hopo_fit_22)[1], label, :red; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

label = L"\Psi^{\mathrm{a}} = \frac{a}{2b}e^{b(I^{\textrm{e}}_{1}-3)} + \frac{a_f}{2b_f}e^{b_f \langle I^{\textrm{e}}_{4,f}-1 \rangle ^2}"
plot_equibiaxial(ax4f, 1,1, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1], label, :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4s, 2,2, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1], label, :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))
plot_equibiaxial(ax4n, 3,3, F -> constitutive_driver(F, f₀, s₀, n₀, 1.0, new_model_hoho_fit)[1], label, :blue; λₘᵢₙ=minimum(new_xdata), λₘₐₓ=maximum(new_xdata))

scatter!(ax4f, linyin1998_4_fiber_active, label="Lin Yin (1998) fig 4 (active)", marker=:xcross)
scatter!(ax4s, linyin1998_4_sheet_active, label="Lin Yin (1998) fig 4 (active)", marker=:xcross)

f4[1,4] = Legend(f4, ax4f, framevisible=false, bgcolor=:white)

save("active_stress_and_mgmk.eps", f4)
