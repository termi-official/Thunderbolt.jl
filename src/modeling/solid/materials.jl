# TODO (FILE) I think we should change the design here. Instea of dispatching on Ψ we should make the material callable or equip it with a function.

abstract type QuasiStaticModel end

function material_routine(constitutive_model::QuasiStaticModel, F, internal_state, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    coefficients = evaluate_coefficient(constitutive_model.microstructure_model, geometry_cache, qp, time)
    return material_routine(F, coefficients, internal_state, constitutive_model)
end

@doc raw"""
    PrestressedMaterialModel(inner_model, prestress_field)
"""
struct PrestressedMaterialModel{MM, FF}
    inner_model::MM
    prestress_field::FF
end

function material_routine(constitutive_model::PrestressedMaterialModel, F, internal_state, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    F₀ = evaluate_coefficient(constitutive_model.prestress_field, geometry_cache, qp, time)
    return material_routine(constitutive_model.inner_model, F ⋅ F₀, internal_state, geometry_cache, qp, time)
end

setup_internal_model_cache(cv, constitutive_model::PrestressedMaterialModel) = setup_internal_model_cache(cv, constitutive_model.inner_model)

@doc raw"""
    PK1Model(material, internal_model, coefficient_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor. If the material is energy-based,
then the term is formulated as follows:
$$\int_{\Omega_0} P(u,s) \cdot \delta F dV = \int_{\Omega_0} \partial_{F} \psi(u,s) \cdot \delta \nabla u $$
"""
struct PK1Model{PMat, IMod, CFType} <: QuasiStaticModel
    material::PMat
    internal_model::IMod
    coefficient_field::CFType
end

setup_internal_model_cache(cv, constitutive_model::PK1Model{<:Any, Nothing}) = EmptyInternalVariableCache()

function material_routine(F::Tensor{2,dim}, coefficients, internal_state::Nothing, model::PK1Model) where {dim}
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     coefficients, model.material)
            + Ψ(F_ad, Fᵃ, coefficients, model.material),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end

@doc raw"""
    GeneralizedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The generalized Hill framework as proposed by [GokMenKuh:2014:ghm](@citet).

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely.
"""
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod, MS} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function material_routine(F::Tensor{2,dim}, coefficients, internal_state, model::GeneralizedHillModel) where {dim}
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(internal_state, coefficients, model.contraction_model, model.active_deformation_gradient_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     coefficients, model.passive_spring)
            + Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ExtendedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The extended (generalized) Hill model as proposed by [OgiBalPer:2023:aeg](@citet). The original formulation dates back to [StaKlaHol:2008:smc](@citet) for smooth muscle tissues.

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + \mathcal{N}(\bm{\alpha})W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely. $\mathcal{N}$ is the amount of formed crossbridges. We refer to the original paper [OgiBalPer:2023:aeg](@cite) for more details.
"""
struct ExtendedHillModel{PMat, AMat, ADGMod, CMod, MS} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function material_routine(F::Tensor{2,dim}, coefficients, cell_state, model::ExtendedHillModel) where {dim}
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(cell_state, coefficients, model.contraction_model, model.active_deformation_gradient_model)
    N = 𝓝(cell_state, model.contraction_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
                Ψ(F_ad,     coefficients, model.passive_spring)
            + N*Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ActiveStressModel(material_model, active_stress_model, contraction_model, microstructure_model)

The active stress model as originally proposed by [GucWalMcC:1993:mac](@citet).

In this framework the model is formulated via balance of linear momentum in the first Piola Kirchhoff $\mathbf{P}$:

$\mathbf{P}(\mathbf{F},T^{\rm{a}}) := \partial_{\mathbf{F}} W_{\rm{passive}}(\mathbf{F}) + \mathbf{P}^{\rm{a}}(\mathbf{F}, T^{\rm{a}})$

where the passive material response can be described by an energy $W_{\rm{passive}$ and $T^{\rm{a}}$ the active tension generated by the contraction model.
"""
struct ActiveStressModel{Mat, ASMod, CMod, MS} <: QuasiStaticModel
    material_model::Mat
    active_stress_model::ASMod
    contraction_model::CMod
    microstructure_model::MS
end

function material_routine(F::Tensor{2,dim}, coefficients, cell_state, model::ActiveStressModel) where {dim}
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad, coefficients, model.material_model),
        F, :all)

    λᵃ = compute_λᵃ(cell_state, model.contraction_model)
    ∂2 = Tensors.gradient(
        F_ad -> ∂(model.active_stress_model, cell_state, F_ad, coefficients),
    F)
    N = 𝓝(cell_state, model.contraction_model)
    return ∂Ψ∂F + N*∂(model.active_stress_model, cell_state, F, coefficients), ∂²Ψ∂F² + N*∂2
end


"""
    ElastodynamicsModel(::QuasiStaticModel, ρ::Coefficient)
"""
struct ElastodynamicsModel{RHSModel <: QuasiStaticModel, CoefficientType}
    rhs::RHSModel
    ρ::CoefficientType
end

setup_internal_model_cache(cv, constitutive_model::Union{<:ActiveStressModel, <:ExtendedHillModel, <:GeneralizedHillModel}) = setup_contraction_model_cache(cv, constitutive_model.contraction_model)
setup_internal_model_cache(cv, constitutive_model::Union{<:ElastodynamicsModel{<:ActiveStressModel}, <:ElastodynamicsModel{<:ExtendedHillModel}, <:ElastodynamicsModel{<:GeneralizedHillModel}}) = setup_contraction_model_cache(cv, constitutive_model.rhs.contraction_model)
