
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
    Fᵃ = compute_Fᵃ(Caᵢ, f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)

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
struct ExtendedHillModel{PMat, AMat, ADGMod, CMod} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
end

"""
"""
function constitutive_driver(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, Caᵢ, model::ExtendedHillModel) where {dim}
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(Caᵢ, f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)
    N = 𝓝(Caᵢ, model.contraction_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
                Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + N*Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
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
        F_ad -> ∂(model.active_stress_model, Caᵢ, F_ad, f₀, s₀, n₀),
    F)
    N = 𝓝(Caᵢ, model.contraction_model)
    return ∂Ψ∂F + N*∂(model.active_stress_model, Caᵢ, F, f₀, s₀, n₀), ∂²Ψ∂F² + N*∂2
end


"""
"""
struct ElastodynamicsModel{RHSModel <: QuasiStaticModel, CoefficienType}
    rhs::RHSModel
    ρ::CoefficienType
    # TODO refactor into solver cache
    vₜ₋₁::Vector
end

"""
"""
struct CardiacMechanicalElementCache{MP, MSCache, CFCache, CMCache, CV}
    mp::MP
    microstructure_cache::MSCache
    calcium_field_cache::CFCache
    # coordinate_system_cache::CSCache
    contraction_model_cache::CMCache
    cv::CV
end

function update_element_cache!(cache::CardiacMechanicalElementCache{MP, MSCache, CMCache, CV}, calcium_field::CF, time::Float64, cell::CellCacheType) where {CellCacheType, MP, MSCache, CMCache, CV, CF}
    reinit!(cache.cv, cell)
    update_microstructure_cache!(cache.microstructure_cache, time, cell, cache.cv)
    update_contraction_model_cache!(cache.contraction_model_cache, time, cell, cache.cv, calcium_field)
end

function assemble_element!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::CardiacMechanicalElementCache)
    # @unpack mp, microstructure_cache, coordinate_system_cache, cv, contraction_model_cache = cache
    @unpack mp, microstructure_cache, contraction_model_cache, cv = cache
    ndofs = getnbasefunctions(cv)

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        f₀, s₀, n₀ = directions(microstructure_cache, qp)
        contraction_state = state(contraction_model_cache, qp)
        # x = coordinate(coordinate_system_cache, qp)
        P, ∂P∂F = constitutive_driver(F, f₀, s₀, n₀, contraction_state, mp)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end
