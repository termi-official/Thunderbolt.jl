
abstract type QuasiStaticModel end

"""
@TODO citation
"""
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
end

"""
"""
function constitutive_driver(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, Caᵢ, model::GeneralizedHillModel) where {dim}
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(Caᵢ,  f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end

"""
"""
struct ActiveStressModel{Mat, ASMod, CMod} <: QuasiStaticModel
    material_model::Mat
    active_stress_model::ASMod
    contraction_model::CMod
end

"""
"""
function constitutive_driver(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, Caᵢ, model::ActiveStressModel) where {dim}
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.material_model),
        F, :all)

    λᵃ = compute_λᵃ(Caᵢ, model.contraction_model)
    ∂2 = Tensors.gradient(
        F_ad -> ∂(model.active_stress_model, λᵃ, Caᵢ, F_ad, f₀, s₀, n₀),
    F)
    return ∂Ψ∂F + ∂(model.active_stress_model, λᵃ, Caᵢ, F, f₀, s₀, n₀), ∂²Ψ∂F² + ∂2 
end


"""
"""
struct ElastodynamicsModel{RHSModel <: QuasiStaticModel, CoefficienType}
    rhs::RHSModel
    ρ::CoefficienType
    # TODO refactor into cache
    vₜ₋₁::Vector
end
