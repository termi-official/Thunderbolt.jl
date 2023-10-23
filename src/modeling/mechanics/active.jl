"""
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

"""
@TODO citation original GMK paper
"""
struct GMKActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SSSM, ::GMKActiveDeformationGradientModel) where {SSSM <: SteadyStateSarcomereModel}
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = Tensors.unsafe_symmetric(one(SymmetricTensor{2, 3}) + (λᵃ - 1.0) * f₀ ⊗ f₀)
    return Fᵃ
end


"""
@TODO citation 10.1016/j.euromechsol.2013.10.009
"""
struct GMKIncompressibleActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SSSM, ::GMKIncompressibleActiveDeformationGradientModel) where {SSSM <: SteadyStateSarcomereModel}
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = λᵃ*f₀ ⊗ f₀ + 1.0/sqrt(λᵃ) * s₀ ⊗ s₀ + 1.0/sqrt(λᵃ) * n₀ ⊗ n₀
    return Fᵃ
end

"""
@TODO citation 10.1016/j.euromechsol.2013.10.009
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

Base.@kwdef struct SimpleActiveStress{TD}
    Tmax::TD = 1.0
end

∂(sas::SimpleActiveStress, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tmax * Caᵢ * (F ⋅ f₀ ) ⊗ f₀ / norm(F ⋅ f₀)

Base.@kwdef struct PiersantiActiveStress{TD}
    Tmax::TD = 1.0
    pf::TD = 1.0
    ps::TD = 0.75
    pn::TD = 0.0
end

∂(sas::PiersantiActiveStress, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tmax * Caᵢ * (sas.pf*(F ⋅ f₀) ⊗ f₀ / norm(F ⋅ f₀) + sas.ps*(F ⋅ s₀) ⊗ s₀ / norm(F ⋅ s₀) + sas.pn * (F ⋅ n₀) ⊗ n₀ / norm(F ⋅ n₀))


Base.@kwdef struct Guccione1993ActiveModel
    Tₘₐₓ::Float64 = 100.0
end

∂(sas::Guccione1993ActiveModel, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tₘₐₓ * Caᵢ * (F ⋅ f₀ ) ⊗ f₀
