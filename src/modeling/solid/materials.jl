# TODO (FILE) I think we should change the design here. Instea of dispatching on Ψ we should make the material callable or equip it with a function.

abstract type QuasiStaticModel end

#TODO constrain to orthotropic material models, e.g. via traits, or rewrite all 3 "material_routine"s below
function material_routine(constitutive_model, F, internal_state, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    f₀, s₀, n₀ = evaluate_coefficient(constitutive_model.microstructure_model, geometry_cache, qp, time)
    return material_routine(F, f₀, s₀, n₀, internal_state, constitutive_model)
end


"""
@TODO citation
"""
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod, MS} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

"""
"""
function material_routine(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, internal_state, model::GeneralizedHillModel) where {dim}
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(internal_state, f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


"""
@TODO citation
"""
struct ExtendedHillModel{PMat, AMat, ADGMod, CMod, MS} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

"""
"""
function material_routine(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, cell_state, model::ExtendedHillModel) where {dim}
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(cell_state, f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)
    N = 𝓝(cell_state, model.contraction_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
                Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + N*Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


"""
"""
struct ActiveStressModel{Mat, ASMod, CMod, MS} <: QuasiStaticModel
    material_model::Mat
    active_stress_model::ASMod
    contraction_model::CMod
    microstructure_model::MS
end

"""
"""
function material_routine(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, cell_state, model::ActiveStressModel) where {dim}
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.material_model),
        F, :all)

    λᵃ = compute_λᵃ(cell_state, model.contraction_model)
    ∂2 = Tensors.gradient(
        F_ad -> ∂(model.active_stress_model, cell_state, F_ad, f₀, s₀, n₀),
    F)
    N = 𝓝(cell_state, model.contraction_model)
    return ∂Ψ∂F + N*∂(model.active_stress_model, cell_state, F, f₀, s₀, n₀), ∂²Ψ∂F² + N*∂2
end


"""
"""
struct ElastodynamicsModel{RHSModel <: QuasiStaticModel, CoefficienType}
    rhs::RHSModel
    ρ::CoefficienType
end
