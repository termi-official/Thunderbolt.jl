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

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SteadyStateSarcomereModel, ::GMKActiveDeformationGradientModel)
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = Tensors.unsafe_symmetric(one(SymmetricTensor{2, 3}) + (λᵃ - 1.0) * f₀ ⊗ f₀)
    return Fᵃ
end


"""
@TODO citation 10.1016/j.euromechsol.2013.10.009
"""
struct GMKIncompressibleActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SteadyStateSarcomereModel, ::GMKIncompressibleActiveDeformationGradientModel)
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = λᵃ*f₀ ⊗ f₀ + 1.0/sqrt(λᵃ) * s₀ ⊗ s₀ + 1.0/sqrt(λᵃ) * n₀ ⊗ n₀
    return Fᵃ
end

"""
@TODO citation 10.1016/j.euromechsol.2013.10.009
"""
struct RLRSQActiveDeformationGradientModel
    sheetlet_part
end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::SteadyStateSarcomereModel, Fᵃ_model::RLRSQActiveDeformationGradientModel)
    @unpack sheetlet_part = Fᵃ_model
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = λᵃ * f₀⊗f₀ + (1.0+sheetlet_part*(λᵃ-1.0))* s₀⊗s₀ + 1.0/((1.0+sheetlet_part*(λᵃ-1.0))*λᵃ) * n₀⊗n₀
    return Fᵃ
end

Base.@kwdef struct SimpleActiveStress
    Tmax = 1.0
end

∂(sas::SimpleActiveStress, λᵃ, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tmax * Caᵢ * λᵃ * (F ⋅ f₀ ) ⊗ f₀

Base.@kwdef struct PiersantiActiveStress
    Tmax = 1.0
    pf = 1.0
    ps = 0.75
    pn = 0.0
end

∂(sas::PiersantiActiveStress, λᵃ, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tmax * Caᵢ * λᵃ * (sas.pf*(F ⋅ f₀) ⊗ f₀ / norm(F ⋅ f₀) + sas.ps*(F ⋅ s₀) ⊗ s₀ / norm(F ⋅ s₀) + sas.pn * (F ⋅ n₀) ⊗ n₀ / norm(F ⋅ n₀))

