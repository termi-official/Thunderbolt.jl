"""
A simple helper to use a passive material model as an active material for [GeneralizedHillModel](@ref), [ExtendedHillModel](@ref) and [ActiveStressModel](@ref).
"""
struct ActiveMaterialAdapter{Mat}
    mat::Mat
end

"""
"""
function Ψ(F, Fᵃ, f₀, s₀, n₀, adapter::ActiveMaterialAdapter)
	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
    s̃ = Fᵃ ⋅ s₀ / norm(Fᵃ ⋅ s₀)
    ñ = Fᵃ ⋅ n₀ / norm(Fᵃ ⋅ n₀)

	Fᵉ = F⋅inv(Fᵃ)

	Ψᵃ = Ψ(Fᵉ, f̃, s̃, ñ, adapter.mat)
    return Ψᵃ
end

@doc raw"""
The active deformation gradient formulation by [GokMenKuh:2014:ghm](@citet).

$F^{\rm{a}} = (\lambda^{\rm{a}}-1) f_0 \otimes f_0$ + I$

See also [OgiBalPer:2023:aeg](@cite) for a further analysis.
"""
struct GMKActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SSSM, ::GMKActiveDeformationGradientModel) where {SSSM <: SteadyStateSarcomereModel}
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = Tensors.unsafe_symmetric(one(SymmetricTensor{2, 3}) + (λᵃ - 1.0) * f₀ ⊗ f₀)
    return Fᵃ
end


@doc raw"""
An incompressivle version of the active deformation gradient formulation by [GokMenKuh:2014:ghm](@citet).

$F^{\rm{a}} = \lambda^{\rm{a}} f_0 \otimes f_0 + \frac{1}{\sqrt{\lambda^{\rm{a}}}}(s_0 \otimes s_0 + n_0 \otimes n_0)$

See also [OgiBalPer:2023:aeg](@cite) for a further analysis.
"""
struct GMKIncompressibleActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SSSM, ::GMKIncompressibleActiveDeformationGradientModel) where {SSSM <: SteadyStateSarcomereModel}
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = λᵃ*f₀ ⊗ f₀ + 1.0/sqrt(λᵃ) * s₀ ⊗ s₀ + 1.0/sqrt(λᵃ) * n₀ ⊗ n₀
    return Fᵃ
end

@doc raw"""
The active deformation gradient formulation by [RosLasRuiSeqQua:2014:tco](@citet).

$F^{\rm{a}} = \lambda^{\rm{a}} f_0 \otimes f_0 + (1+\kappa(\lambda^{\rm{a}}-1)) s_0 \otimes s_0 + \frac{1}{1+\kappa(\lambda^{\rm{a}}-1))\lambda^{\rm{a}}} n_0 \otimes n_0$

Where $\kappa \geq 0$ is the sheelet part.

See also [OgiBalPer:2023:aeg](@cite) for a further analysis.
"""
struct RLRSQActiveDeformationGradientModel{TD}
    sheetlet_part::TD
end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SSSM, Fᵃ_model::RLRSQActiveDeformationGradientModel) where {SSSM <: SteadyStateSarcomereModel}
    @unpack sheetlet_part = Fᵃ_model
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = λᵃ * f₀⊗f₀ + (1.0+sheetlet_part*(λᵃ-1.0))* s₀⊗s₀ + 1.0/((1.0+sheetlet_part*(λᵃ-1.0))*λᵃ) * n₀⊗n₀
    return Fᵃ
end


@doc raw"""
A simple active stress component.

$T^{\rm{a}} = T^{\rm{max}} \, [Ca_{\rm{i}}] \frac{(F \cdot f_0) \otimes f_0}{||F \cdot f_0||}$
"""
Base.@kwdef struct SimpleActiveStress{TD}
    Tmax::TD = 1.0
end

∂(sas::SimpleActiveStress, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tmax * Caᵢ * (F ⋅ f₀ ) ⊗ f₀ / norm(F ⋅ f₀)


@doc raw"""
The active stress component described by [PieRegSalCorVerQua:2022:clm](@citet) (Eq. 3).

$T^{\rm{a}} = T^{\rm{max}} \, [Ca_{\rm{i}}] \left(p^f \frac{(F \cdot f_0) \otimes f_0}{||F \cdot f_0||} + p^{\rm{s}} \frac{(F \cdot s_0) \otimes s_0}{||F \cdot s_0||} + p^{\rm{n}} \frac{(F \cdot n_0) \otimes n_0}{||F \cdot n_0||}\right)$
"""
Base.@kwdef struct PiersantiActiveStress{TD}
    Tmax::TD = 1.0
    pf::TD = 1.0
    ps::TD = 0.75
    pn::TD = 0.0
end

∂(sas::PiersantiActiveStress, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tmax * Caᵢ * (sas.pf*(F ⋅ f₀) ⊗ f₀ / norm(F ⋅ f₀) + sas.ps*(F ⋅ s₀) ⊗ s₀ / norm(F ⋅ s₀) + sas.pn * (F ⋅ n₀) ⊗ n₀ / norm(F ⋅ n₀))


@doc raw"""
The active stress component as described by [GucWalMcC:1993:mac](@citet).

$T^{\rm{a}} = T^{\rm{max}} \, [Ca_{\rm{i}}] (F \cdot f_0) \otimes f_0$

"""
Base.@kwdef struct Guccione1993ActiveModel
    Tₘₐₓ::Float64 = 100.0
end

∂(sas::Guccione1993ActiveModel, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tₘₐₓ * Caᵢ * (F ⋅ f₀ ) ⊗ f₀
