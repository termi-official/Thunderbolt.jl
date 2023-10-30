
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
end

"""
"""
struct CardiacMechanicalElementCache{MP, MSCache, CMCache, CV}
    mp::MP
    microstructure_cache::MSCache
    # coordinate_system_cache::CSCache
    contraction_model_cache::CMCache
    cv::CV
end

function assemble_element!(Kₑ::Matrix, residualₑ, uₑ, cell, element_cache::CardiacMechanicalElementCache, time)
    @unpack mp, microstructure_cache, contraction_model_cache, cv = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, cell)
    update_microstructure_cache!(microstructure_cache, time, cell, cv)
    update_contraction_model_cache!(contraction_model_cache, time, cell, cv)

    @inbounds for qpᵢ in 1:getnquadpoints(cv)
        ξ = cv.qr.points[qpᵢ]
        qp = QuadraturePoint(qpᵢ, ξ)
        dΩ = getdetJdV(cv, qpᵢ)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qpᵢ, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        f₀, s₀, n₀ = directions(microstructure_cache, qp) # TODO this can be treated as a coefficient inside the constitutive_driver call?
        contraction_state = state(contraction_model_cache, qp)
        P, ∂P∂F = constitutive_driver(F, f₀, s₀, n₀, contraction_state, mp)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qpᵢ, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qpᵢ, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end
