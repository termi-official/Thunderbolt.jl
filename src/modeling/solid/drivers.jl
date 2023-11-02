
struct StructuralModel{MM, FM}
    mechanical_model::MM
    face_models::FM
end

abstract type QuasiStaticModel end

#TODO constrain to orthotropic material models, e.g. via traits, or rewrite all 3 "constitutive_driver"s below
function constitutive_driver(constitutive_model, F, internal_state, geometry_cache, qp::QuadraturePoint, time)
    f₀, s₀, n₀ = evaluate_coefficient(constitutive_model.microstructure_model, geometry_cache, qp, time)
    return constitutive_driver(F, f₀, s₀, n₀, internal_state, constitutive_model)
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
function constitutive_driver(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, internal_state, model::GeneralizedHillModel) where {dim}
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
function constitutive_driver(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, cell_state, model::ExtendedHillModel) where {dim}
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
function constitutive_driver(F::Tensor{2,dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}, cell_state, model::ActiveStressModel) where {dim}
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

"""
"""
struct StructuralElementCache{M, CMCache, CV}
    constitutive_model::M
    contraction_model_cache::CMCache
    cv::CV
end

function assemble_element!(Kₑ::Matrix, residualₑ, uₑ, geometry_cache, element_cache::StructuralElementCache, time)
    @unpack constitutive_model, contraction_model_cache, cv = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qpᵢ in 1:getnquadpoints(cv)
        ξ = cv.qr.points[qpᵢ]
        qp = QuadraturePoint(qpᵢ, ξ)
        dΩ = getdetJdV(cv, qpᵢ)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qpᵢ, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        contraction_state = state(contraction_model_cache, geometry_cache, qp, time)
        P, ∂P∂F = constitutive_driver(constitutive_model, F, contraction_state, geometry_cache, qp, time)

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
