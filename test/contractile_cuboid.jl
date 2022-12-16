using Ferrite, SparseArrays, BlockArrays, Thunderbolt, UnPack, Tensors

struct NullEnergyModel end
Ψ(F, f₀, s₀, n₀, mp::NullEnergyModel) = 0.0

# TODO citation
@Base.kwdef struct NeffCompressionPenalty
    a  = 1.0
    b  = 2.0
    β  = 1.0
end
function U(I₃, mp::NeffCompressionPenalty)
    mp.β * (I₃^mp.b + 1/I₃^mp.b - 2)^mp.a
end

# TODO how is this one called in literature? citation?
@Base.kwdef struct SimpleCompressionPenalty
    β  = 1.0
end
function U(I₃, mp::SimpleCompressionPenalty)
    mp.β * (I₃ - 1 - 2*log(sqrt(I₃)))
end

# https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
@Base.kwdef struct TransverseIsotopicNeoHookeanModel
	a₁ = 2.6
	a₂ = 2.82
	α₁ = 30.48
	α₂ = 7.25

    mpU = NeffCompressionPenalty()
end

"""
"""
function Ψ(F, f₀, s₀, n₀, mp::TransverseIsotopicNeoHookeanModel)
    @unpack a₁, a₂, α₁, α₂, mpU = mp
	C = tdot(F)
    I₁ = tr(C)
	I₃ = det(C)

    Ī₁ = I₁/cbrt(I₃)
    # this is a hotfix to fight numerical noise when returning to the equilibrium state...
    if -1e-8 < Ī₁ - 3.0 < 0.0
        Ī₁ = 3.0
    end

	I₄ = tr(C ⋅ f₀ ⊗ f₀)

	Ψᵖ = α₁*(Ī₁ - 3)^a₁ + U(I₃, mpU)
    if I₄ > 1
        Ψᵖ += α₂*(I₄ - 1)^2
    end

    return Ψᵖ
end

# TODO citation
Base.@kwdef struct HolzapfelOgden2009Model #<: OrthotropicMaterialModel
	a   =  0.059
	b   =  8.023
	aᶠ  = 18.472
	bᶠ  = 16.026
	aˢ  =  2.581
	bˢ  = 11.120
	aᶠˢ =  0.216
	bᶠˢ = 11.436
	mpU = SimpleCompressionPenalty()
end

function Ψ(F, f₀, s₀, n₀, mp::HolzapfelOgden2009Model)
	# Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
    @unpack a, b, aᶠ, bᶠ, aˢ, bˢ, aᶠˢ, bᶠˢ, mpU = mp

	C = tdot(F)
	I₃ = det(C)
	J = det(F)
    I₁ = tr(C/cbrt(J^2))
	I₄ᶠ = f₀ ⋅ C ⋅ f₀
	I₄ˢ = s₀ ⋅ C ⋅ s₀
	I₈ᶠˢ = (f₀ ⋅ C ⋅ s₀ + s₀ ⋅ C ⋅ f₀)/2.0

	Ψᵖ = a/(2.0*b)*exp(b*(I₁-3.0)) + aᶠˢ/(2.0*bᶠˢ)*(exp(bᶠˢ*I₈ᶠˢ^2)-1.0) + U(I₃, mpU)
	if I₄ᶠ > 1.0
		Ψᵖ += aᶠ/(2.0*bᶠ)*(exp(bᶠ*(I₄ᶠ - 1)^2)-1.0)
	end
	if I₄ˢ > 1.0
		Ψᵖ += aˢ/(2.0*bˢ)*(exp(bˢ*(I₄ˢ - 1)^2)-1.0)
	end

    return Ψᵖ
end


"""
"""
Base.@kwdef struct LinYinPassiveModel #<: TransverseIsotropicMaterialModel
	C₁ = 1.05
	C₂ = 9.13
	C₃ = 2.32
	C₄ = 0.08
    mpU = SimpleCompressionPenalty()
end

"""
"""
function Ψ(F, f₀, s₀, n₀, model::LinYinPassiveModel)
	@unpack C₁, C₂, C₃, C₄, mpU = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	I₃ = det(C)
	I₄ = f₀ ⋅ C ⋅ f₀ # = C : (f ⊗ f)

	# Exponential portion
	Q = C₂*(I₁-3)^2 + C₃*(I₁-3)*(I₄-1) + C₄*(I₄-1)^2
	return C₁*(exp(Q)-1) + U(I₃, mpU)
end

Base.@kwdef struct LinYinActiveModel #<: TransverseIsotropicMaterialModel
    C₀ = 0.0
	C₁ = -13.03
	C₂ = 36.65
	C₃ = 35.42
	C₄ = 15.52
    C₅ = 1.62
    mpU = SimpleCompressionPenalty()
end

"""
"""
function Ψ(F, f₀, s₀, n₀, model::LinYinActiveModel)
	@unpack C₀, C₁, C₂, C₃, C₄, C₅, mpU = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	I₃ = det(C)
	I₄ = f₀ ⋅ C ⋅ f₀ # = C : (f ⊗ f)

	return C₀ + C₁*(I₁-3)*(I₄-1) + C₂*(I₁-3)^2 + C₃*(I₄-1)^2 + C₄*(I₁-3) + C₅*(I₄-1) + U(I₃, mpU)
end


Base.@kwdef struct HumphreyStrumpfYinModel #<: TransverseIsotropicMaterialModel
	C₁ = 15.93
	C₂ = 55.85
	C₃ =  3.59
	C₄ = 30.21
    mpU = SimpleCompressionPenalty()
end

"""
"""
function Ψ(F, f₀, s₀, n₀, model::HumphreyStrumpfYinModel)
	@unpack C₁, C₂, C₃, C₄, mpU = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	I₃ = det(C)
	I₄ = f₀ ⋅ C ⋅ f₀ # = C : (f ⊗ f)

	return C₁*(√I₄-1)^2 + C₂*(√I₄-1)^3 + C₃*(√I₄-1)*(I₁-3) + C₄*(I₁-3)^2 + U(I₃, mpU)
end

@Base.kwdef struct LinearSpringModel
	η = 10.0
end
function Ψ(F, f₀, s₀, n₀, mp::LinearSpringModel)
    @unpack η = mp

    M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
	FMF = Tensors.unsafe_symmetric(F ⋅ M ⋅ transpose(F))
	I₄ = tr(FMF)

    return η / 2.0 * (I₄ - 1)^2.0
end

struct ActiveMaterialAdapter{Mat}
    mat::Mat
end

function Ψ(F, Fᵃ, f₀, s₀, n₀, adapter::ActiveMaterialAdapter)
	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
    s̃ = Fᵃ ⋅ s₀ / norm(Fᵃ ⋅ s₀)
    ñ = Fᵃ ⋅ n₀ / norm(Fᵃ ⋅ n₀)

	Fᵉ = F⋅inv(Fᵃ)

	Ψᵃ = Ψ(Fᵉ, f̃, s̃, ñ, adapter.mat)
    return Ψᵃ
end

"""
@TODO citation pelce paper
"""
Base.@kwdef struct PelceSunLangeveld1995Model 
    β = 3.0
    λᵃₘₐₓ = 0.7
end

function compute_λᵃ(Ca, mp::PelceSunLangeveld1995Model)
    @unpack β, λᵃₘₐₓ = mp
    f(c) = c > 0.0 ? 0.5 + atan(β*log(c))/π  : 0.0
    return 1.0 / (1.0 + f(Ca)*(1.0/λᵃₘₐₓ - 1.0))
end

"""
@TODO citation original GMK paper
"""
struct GMKActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::PelceSunLangeveld1995Model, ::GMKActiveDeformationGradientModel)
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = Tensors.unsafe_symmetric(one(SymmetricTensor{2, 3}) + (λᵃ - 1.0) * f₀ ⊗ f₀)
    return Fᵃ
end

"""
@TODO citation 10.1016/j.euromechsol.2013.10.009
"""
struct GMKIncompressibleActiveDeformationGradientModel end

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::PelceSunLangeveld1995Model, ::GMKIncompressibleActiveDeformationGradientModel)
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

function compute_Fᵃ(Ca, f₀, s₀, n₀, contraction_model::PelceSunLangeveld1995Model, Fᵃ_model::RLRSQActiveDeformationGradientModel)
    @unpack sheetlet_part = Fᵃ_model
    λᵃ = compute_λᵃ(Ca, contraction_model)
    Fᵃ = λᵃ * f₀⊗f₀ + (1.0+sheetlet_part*(λᵃ-1.0))* s₀⊗s₀ + 1.0/((1.0+sheetlet_part*(λᵃ-1.0))*λᵃ) * n₀⊗n₀
    return Fᵃ
end

abstract type QuasiStaticModel end
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod} <: QuasiStaticModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
end

struct ActiveStressModel{Mat, AMat, ADGMod, CMod} <: QuasiStaticModel
    material_model::Mat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
end

struct ConstantFieldCoefficient{T}
    val::T
end
value(coeff::ConstantFieldCoefficient, cell_id, ξ, t = 0.0) = coeff.val

struct FieldDataCoefficient
	elementwise_data #3d array (element_idx, base_fun_idx, dim)
	ip
end
function value(coeff::FieldDataCoefficient, cell_id, ξ, t = 0.0)
	@unpack ip, elementwise_data = coeff
	dim = 3 #@FIXME PLS

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{dim, Float64})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.value(ip, i, ξ) * elementwise_data[cell_id, i]
    end
    return val / norm(val)
end

struct ElastodynamicsModel{RHSModel <: QuasiStaticModel, CoefficienType}
    rhs::RHSModel
    ρ::CoefficienType
    # TODO refactor into cache
    vₜ₋₁::Vector
end

struct OrthotropicMicrostructureModel{FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
    normal_coefficient::NormalCoefficientType
end

function directions(fsn::OrthotropicMicrostructureModel, cell_id, ξ, t = 0.0)
    f₀ = value(fsn.fiber_coefficient, cell_id, ξ, t)
    s₀ = value(fsn.sheetlet_coefficient, cell_id, ξ, t)
    n₀ = value(fsn.normal_coefficient, cell_id, ξ, t)

    f₀, s₀, n₀
end

function constitutive_driver(F, f₀, s₀, n₀, Caᵢ, model::GeneralizedHillModel)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(Caᵢ,  f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


function constitutive_driver(F, f₀, s₀, n₀, Caᵢ, model::ActiveStressModel)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(Caᵢ,  f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


function constitutive_driver_rhs(F, f₀, s₀, n₀, Caᵢ, model::GeneralizedHillModel)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(Caᵢ,  f₀, s₀, n₀, model.contraction_model, model.active_deformation_gradient_model)

    ∂Ψ∂F = Tensors.gradient(
        F_ad ->
              Ψ(F_ad,     f₀, s₀, n₀, model.passive_spring)
            + Ψ(F_ad, Fᵃ, f₀, s₀, n₀, model.active_spring),
        F)

    return ∂Ψ∂F
end


function assemble_element_rhs!(
    residualₑ, uₑ, # Element local quantities
    cell, # Ferrite cell
    cv, fv, # Ferrite FEValues
    mp, fsn_model, # Model Parts
    time) # Time point

    cellid = cell.current_cellid.x

	Caᵢ(cellid,x,t) = t < 1.0 ? t : 2.0-t
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(residualₑ, 0.0)

    ndofs = getnbasefunctions(cv)

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
		ξ = cv.qr.points[qp]
		f₀, s₀, n₀ = directions(fsn_model, cellid, ξ, time)
		# P = constitutive_driver_rhs(F, f₀, s₀, n₀, Caᵢ(cellid, ξ, time), mp)
        P, ∂P∂F = constitutive_driver(F, f₀, s₀, n₀, Caᵢ(cellid, ξ, time), mp)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end


function assemble_element!(
    Kₑ, residualₑ, uₑ, # Element local quantities
    cell, # Ferrite cell
    cv, fv, # Ferrite FEValues
    mp, fsn_model, # Model Parts
    time) # Time point
	# TODO factor out
	kₛ = 3.5 # "Spring stiffness"
	kᵇ = 0.0 # Basal bending penalty

    cellid = cell.current_cellid.x

	Caᵢ(cellid,x,t) = t < 1.0 ? t : 2.0-t
	#p = 7.5*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
	#p = 5.0*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
	#p = 0.0*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
	p = 0.0

    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(Kₑ, 0.0)
    fill!(residualₑ, 0.0)

    ndofs = getnbasefunctions(cv)

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
		ξ = cv.qr.points[qp]
		f₀, s₀, n₀ = directions(fsn_model, cellid, ξ, time)
		P, ∂P∂F = constitutive_driver(F, f₀, s₀, n₀, Caᵢ(cellid, ξ, time), mp)

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

    # Surface integrals
    # for local_face_index in 1:nfaces(cell)
	# 	# How does this interact with the stress?
	# 	if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Epicardium")
    #         reinit!(fv, cell, local_face_index)
	# 		ndofs_face = getnbasefunctions(fv)
    #         for qp in 1:getnquadpoints(fv)
    #             dΓ = getdetJdV(fv, qp)

	# 			# ∇u_prev = function_gradient(fv, qp, uₑ_prev)
    #             # F_prev = one(∇u_prev) + ∇u_prev
	# 			# N = transpose(inv(F_prev)) ⋅ getnormal(fv, qp) # TODO this may mess up reversibility

	# 			N = getnormal(fv, qp)

	# 			u_q = function_value(fv, qp, uₑ)
	# 			#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.0, u_q, :all)
	# 			#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

	# 			# Add contribution to the residual from this test function
	# 			for i in 1:ndofs_face
	# 	            δuᵢ = shape_value(fv, qp, i)
	# 				#residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ
	# 				residualₑ[i] += kₛ * δuᵢ ⋅ u_q * dΓ # kₛ = α in Barbarotta et al. (2018)

	# 	            for j in 1:ndofs_face
	# 	                δuⱼ = shape_value(fv, qp, j)
	# 	                # Add contribution to the tangent
	# 	                #Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
	# 					Kₑ[i, j] += kₛ * ( δuᵢ ⋅ δuⱼ ) * dΓ
	# 	            end
	# 			end

	# 			# N = getnormal(fv, qp)
	# 			# u_q = function_value(fv, qp, uₑ)
	# 			# for i ∈ 1:ndofs
	# 			# 	δuᵢ = shape_value(fv, qp, i)
	# 			# 	residualₑ[i] += 0.5 * kₛ * (δuᵢ ⋅ N) * (N ⋅ u_q) * dΓ
	# 			# 	for j ∈ 1:ndofs
	# 			# 		δuⱼ = shape_value(fv, qp, j)
	# 			# 		Kₑ[i,j] += 0.5 * kₛ * (δuᵢ ⋅ N) * (N ⋅ δuⱼ) * dΓ
	# 			# 	end
	# 			# end
    #         end
    #     end

	# 	if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Base")
	# 		reinit!(fv, cell, local_face_index)
	# 		ndofs_face = getnbasefunctions(fv)
    #         for qp in 1:getnquadpoints(fv)
    #             dΓ = getdetJdV(fv, qp)
	# 			N = getnormal(fv, qp)

	# 			∇u = function_gradient(fv, qp, uₑ)
    #     		F = one(∇u) + ∇u

	# 			#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(F_ -> 0.0, F, :all)
	# 			∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

	# 			# Add contribution to the residual from this test function
	# 			for i in 1:ndofs_face
	# 	            ∇δui = shape_gradient(fv, qp, i)
	# 				residualₑ[i] += ∇δui ⊡ ∂Ψ∂F * dΓ

	# 	            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
	# 	            for j in 1:ndofs_face
	# 	                ∇δuj = shape_gradient(fv, qp, j)
	# 	                # Add contribution to the tangent
	# 	                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
	# 	            end
	# 			end
	# 		end
	# 	end

	# 	# Pressure boundary
	# 	if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Endocardium")
	# 		reinit!(fv, cell, local_face_index)
	# 		ndofs_face = getnbasefunctions(fv)
    #         for qp in 1:getnquadpoints(fv)
    #             dΓ = getdetJdV(fv, qp)

	# 			#∇u_prev = function_gradient(fv, qp, uₑ_prev)
    #             #F_prev = one(∇u_prev) + ∇u_prev
	# 			#N = transpose(inv(F_prev)) ⋅ getnormal(fv, qp) # TODO this may mess up reversibility

	# 			N = getnormal(fv, qp)

	# 			∇u_q = function_gradient(fv, qp, uₑ)
	# 			F = one(∇u_q) + ∇u_q
	# 			J = det(F)
	# 			cofF = inv(F')
	# 			#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.0, u_q, :all)
	# 			# ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

	# 			# Add contribution to the residual from this test function
	# 			for i in 1:ndofs_face
	# 	            δuᵢ = shape_value(fv, qp, i)
	# 				#residualₑ[i] += p * δuᵢ ⋅ N * dΓ
	# 				residualₑ[i] += p * J * cofF ⋅ N ⋅ δuᵢ * dΓ

	# 	            for j in 1:ndofs_face
	# 	                #δuⱼ = shape_value(fv, qp, j)
	# 					∇δuⱼ = shape_gradient(fv, qp, j)
	# 	                # Add contribution to the tangent
	# 	                #Kₑ[i, j] += ( 0.0 ) * dΓ
	# 					Kₑ[i, j] += p * δuᵢ ⋅ (cofF ⊡ ∇δuⱼ * one(F) - cofF ⋅ ∇δuⱼ') ⋅ (J * cofF) ⋅ N  * dΓ
	# 	            end
	# 			end
    #         end
	# 	end
    # end
end

function assemble_global!(K, f, uₜ, dh, cv, fv, constitutive_model, fsn_model, t)
    n = ndofs_per_cell(dh)
    kₑ = zeros(n, n)
    gₑ = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    #@timeit "assemble" for cell in CellIterator(dh)
	for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        uₑ = uₜ[global_dofs] # element dofs
		assemble_element!(kₑ, gₑ, uₑ, cell, cv, fv, constitutive_model, fsn_model, t)
        assemble!(assembler, global_dofs, gₑ, kₑ)
    end
end

function assemble_rhs!(g, uₜ, dh, cv, fv, constitutive_model, fsn_model, t)
    g .= 0.0
    n = ndofs_per_cell(dh)
    gₑ = zeros(n)
    kₑ = zeros(n, n)

    # Loop over all cells in the grid
    #@timeit "assemble" for cell in CellIterator(dh)
	for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        uₑ = uₜ[global_dofs] # element dofs
		assemble_element_rhs!(gₑ, uₑ, cell, cv, fv, constitutive_model, fsn_model, t)
        assemble!(g, global_dofs, gₑ)
    end
end

function assemble_mass!(cellvalues::CellVectorValues{dim}, M::SparseMatrixCSC, dh::DofHandler, coeff, t) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)

    assembler_M = start_assemble(M)

    #Now we iterate over all cells of the grid
    @inbounds for cell in CellIterator(dh)
        fill!(Me, 0)
        #get the coordinates of the current cell
        coords = getcoordinates(cell)
        cellid = cell.current_cellid.x

        Ferrite.reinit!(cellvalues, cell)
        #loop over all Gauss points
        for q_point in 1:getnquadpoints(cellvalues)
            #get the spatial coordinates of the current gauss point
            #ξ = spatial_coordinate(cellvalues, q_point, coords)
            ξ = cellvalues.qr.points[q_point]
            dΩ = getdetJdV(cellvalues, q_point)
            coeff_qp = 1.0# value(coeff, cellid, ξ, t)
            for i in 1:n_basefuncs
                Nᵢ = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    Nⱼ = shape_value(cellvalues, q_point, j)
                    #mass matrices
                    Me[i,i] += coeff_qp * Nᵢ ⋅ Nⱼ * dΩ
                end
            end
        end

        assemble!(assembler_M, celldofs(cell), Me)
    end
    return M
end;

# TODO this needs quite a bit of refactoring.
# Basically this is a
#     * "NonlinearProblem"
#     * "NewtonRaphsonSolver"
function solve_timestep!(uₜ, uₜ₋₁, dh, dbcs, cv, fv, constitutive_model::ConstitutiveModel, fsn_model, t, Δt) where {ConstitutiveModel <: QuasiStaticModel}
    # Update with new boundary conditions (if available)
    Ferrite.update!(dbcs, t)
    apply!(uₜ, dbcs)

    NEWTON_TOL = 1e-8
    MAX_NEWTON_ITER = 100

    _ndofs = ndofs(dh)

    # TODO move into some kind of cache
    Δu = zeros(_ndofs)
    g  = zeros(_ndofs)

    # TODO move into some kind of cache
    K = create_sparsity_pattern(dh)

    # Perform Newton iterations
    newton_itr = -1
    while true
        newton_itr += 1

        assemble_global!(K, g, uₜ, dh, cv, fv, constitutive_model, fsn_model, t)
        normg = norm(g[Ferrite.free_dofs(dbcs)])
        apply_zero!(K, g, dbcs)
        @info "||g|| = " normg

        if normg < NEWTON_TOL
            break
        elseif newton_itr > MAX_NEWTON_ITER
            error("Reached maximum Newton iterations. Aborting.")
        end

        # @info det(K)
        Δu .= K \ g

        apply_zero!(Δu, dbcs)

        uₜ .-= Δu # Current guess
    end
end

# TODO this needs quite a bit of refactoring.
# Basically this combines
#     * "SemiImplicitEuler"
#     * "NonlinearProblem"
#     * "NewtonRaphsonSolver"
function solve_timestep!(uₜ, uₜ₋₁, dh, dbcs, cv, fv, constitutive_model::ElastodynamicsModel, fsn_model, t, Δt)
    # Update with new boundary conditions (if available)
    Ferrite.update!(dbcs, t)
    # apply!(uₜ, dbcs)

    # vₜ = uₜ₋₁ + Δt*uₜ

    NEWTON_TOL = 1e-8
    MAX_NEWTON_ITER = 100

    _ndofs = ndofs(dh)

    # TODO move into some kind of cache
    Δu = zeros(_ndofs)
    g  = zeros(_ndofs)

    # TODO move into some kind of cache
    M = create_sparsity_pattern(dh)
    cv_mass = CellVectorValues(QuadratureRule{3, RefCube}(1), Lagrange{3, RefCube, 1}())
    assemble_mass!(cv_mass, M, dh, constitutive_model.ρ, t)

    K = create_sparsity_pattern(dh)

    # Perform Newton iterations
    # newton_itr = -1
    # while true
    #     newton_itr += 1

    #     assemble_global!(K, g, uₜ, dh, cv, fv, constitutive_model.rhs, fsn_model, t)

    #     g += M*(vₜ - constitutive_model.vₜ₋₁)

    #     normg = norm(g[Ferrite.free_dofs(dbcs)])
    #     apply_zero!(K, g, dbcs)
    #     @info "||g|| = " normg

    #     if normg < NEWTON_TOL
    #         break
    #     elseif newton_itr > MAX_NEWTON_ITER
    #         error("Reached maximum Newton iterations. Aborting.")
    #     end

    #     # @info det(K)
    #     Δu .= K \ g

    #     apply_zero!(Δu, dbcs)

    #     uₜ .-= Δu # Current guess
    # end

    # assemble_global!(K, g, uₜ₋₁, dh, cv, fv, constitutive_model.rhs, fsn_model, t)
    assemble_rhs!(g, uₜ₋₁, dh, cv, fv, constitutive_model.rhs, fsn_model, t)
    # @info Matrix(M) 
    aₜ₋₁ = M \ -g
    # @info aₜ₋₁

    uₜ .= uₜ₋₁ + Δt * constitutive_model.vₜ₋₁ + Δt^2/2.0 * aₜ₋₁
    # @info uₜ

    # assemble_global!(K, g, uₜ, dh, cv, fv, constitutive_model.rhs, fsn_model, t)
    assemble_rhs!(g, uₜ, dh, cv, fv, constitutive_model.rhs, fsn_model, t)
    aₜ = M \ -g
    vₜ = constitutive_model.vₜ₋₁ + Δt * (aₜ₋₁ + aₜ) / 2.0
    # @info vₜ
    constitutive_model.vₜ₋₁ .= vₜ
end


function solve(constitutive_model, name)
    # grid = generate_grid(Hexahedron, (15, 3, 3), Vec{3}((0.0,0.0,0.0)), Vec{3}((0.5, 0.1, 0.1)))
    grid = generate_grid(Hexahedron, (10, 10, 2), Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0, 1.0, 0.2)))

    dim = 3
    order = 1
    qorder = 2
    Δt = 0.1
    T = 1.0
    ip_geo = Lagrange{dim, RefCube, order}()
    ip = Lagrange{dim, RefCube, order}()
    qr = QuadratureRule{dim, RefCube}(qorder)
    qr_face = QuadratureRule{dim-1, RefCube}(qorder)

    cv = CellVectorValues(qr, ip, ip_geo)
    fv = FaceVectorValues(qr_face, ip, ip_geo)

    # TODO get this from the models
    dim_ionic_state = 1
    internal_state_ionic = Vector(undef, getncells(grid)*length(getpoints(qr))*dim_ionic_state)
    dim_concentration_state = 1
    internal_state_concentration = Vector(undef, getncells(grid)*length(getpoints(qr))*dim_concentration_state)
    dim_contraction_state = 0
    internal_state_contraction = Vector(undef, getncells(grid)*length(getpoints(qr))*dim_contraction_state)

    dh = DofHandler(grid)
    push!(dh, :d, dim, ip)
    close!(dh);
    _ndofs = ndofs(dh)

    # f₀data = Vector{Vec{3}}(undef, getncells(grid))
    # for cellindex ∈ 1:getncells(grid)
    #     f₀data[cellindex] = Vec{3}((1.0, 0.0, 0.0))
    # end
    # s₀data = Vector{Vec{3}}(undef, getncells(grid))
    # for cellindex ∈ 1:getncells(grid)
    #     s₀data[cellindex] = Vec{3}((0.0, 1.0, 0.0))
    # end
    # n₀data = Vector{Vec{3}}(undef, getncells(grid))
    # for cellindex ∈ 1:getncells(grid)
    #     n₀data[cellindex] = Vec{3}((0.0, 0.0, 1.0))
    # end
    # ip_fiber = DiscontinuousLagrange{dim, RefCube, 0}()
    # fsn_model = OrthotropicMicrostructureModel(
    #     FieldCoefficient(f₀data, ip_fiber),
    #     FieldCoefficient(s₀data, ip_fiber),
    #     FieldCoefficient(n₀data, ip_fiber)
    # )
    fsn_model = OrthotropicMicrostructureModel(
        ConstantFieldCoefficient(Vec{3}((1.0, 0.0, 0.0))),
        ConstantFieldCoefficient(Vec{3}((0.0, 1.0, 0.0))),
        ConstantFieldCoefficient(Vec{3}((0.0, 0.0, 1.0)))
    )

    # constitutive_model = 
    # #ElastodynamicsModel(
    #     GeneralizedHillModel(
    #         TransverseIsotopicNeoHookeanModel(),
    #         ActiveMaterialAdapter(LinearSpringModel()),
    #         RLRSQActiveDeformationGradientModel(0.2),
    #         #GMKActiveDeformationGradientModel(),
    #         # GMKIncompressibleActiveDeformationGradientModel(),
    #         PelceSunLangeveld1995Model()
    #     )#,
    # #    ConstantFieldCoefficient(1.0),
    # #    zeros(_ndofs)
    # #)

    #
    # Boundary conditions are added to the problem in the usual way.
    # Please check out the other examples for an in depth explanation.
    # Here we force the extracellular porential to be zero at the boundary.
    dbcs = ConstraintHandler(dh)
    # Clamp three sides
    dbc = Dirichlet(:d, getfaceset(grid, "left"), (x,t) -> [0.0], [1])
    add!(dbcs, dbc)
    dbc = Dirichlet(:d, getfaceset(grid, "front"), (x,t) -> [0.0], [2])
    add!(dbcs, dbc)
    dbc = Dirichlet(:d, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:d, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)

    uₜ   = zeros(_ndofs)
    uₜ₋₁ = zeros(_ndofs)

    pvd = paraview_collection("$name.pvd");
    for t ∈ 0.0:Δt:T
        @info "t = " t

        # Store last solution
        uₜ₋₁ .= uₜ

        solve_timestep!(uₜ, uₜ₋₁, dh, dbcs, cv, fv, constitutive_model, fsn_model, t, Δt)

        # Save the solution
        vtk_grid("$name-$t.vtu", dh) do vtk
            vtk_point_data(vtk,dh,uₜ)
            vtk_save(vtk)
            pvd[t] = vtk
        end
    end
    vtk_save(pvd);
end

# Reproducer for original paper
gmk2014model = GeneralizedHillModel(
    HolzapfelOgden2009Model(),
    ActiveMaterialAdapter(LinearSpringModel()),
    GMKActiveDeformationGradientModel(),
    PelceSunLangeveld1995Model()
)

ppek2017likemodel = GeneralizedHillModel(
    HumphreyStrumpfYinModel(),
    ActiveMaterialAdapter(LinYinActiveModel()),
    GMKIncompressibleActiveDeformationGradientModel(),
    PelceSunLangeveld1995Model()
)

rlrsqlikemodel = GeneralizedHillModel(
    NullEnergyModel(),
    ActiveMaterialAdapter(HolzapfelOgden2009Model()),
    RLRSQActiveDeformationGradientModel(0.3),
    PelceSunLangeveld1995Model()
)
