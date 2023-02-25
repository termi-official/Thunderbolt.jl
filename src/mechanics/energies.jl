# This file contains some common energies.

# abstract type AbstractMaterialModel end

# abstract type IsotropicMaterialModel <: AbstractMaterialModel end

# abstract type TransverseIsotropicMaterialModel <: AbstractMaterialModel end

# abstract type OrthotropicMaterialModel <: AbstractMaterialModel end

struct NullEnergyModel end
Ψ(F, f₀, s₀, n₀, mp::NullEnergyModel) = 0.0

struct NullCompressionModel end
U(I₃, mp::NullCompressionModel) = 0.0

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
	if I₄ᶠ >= 1.0
		Ψᵖ += aᶠ/(2.0*bᶠ)*(exp(bᶠ*(I₄ᶠ - 1)^2)-1.0)
	end
	if I₄ˢ >= 1.0
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


# """
# """
# Base.@kwdef struct BarbarottaRossiDedeQuarteroni2018 #<: OrthotropicMaterialModel
# 	a   =  0.2
# 	b   =  4.614
# 	aᶠ  =  4.1907
# 	bᶠ  =  7.8565
# 	aˢ  =  2.564
# 	bˢ  = 10.446
# 	aᶠˢ =  0.1304
# 	bᶠˢ = 15.255
# 	K   = 1.000 # Value not stated in paper
# 	k   = 4.0
# end

# """
# """
# function Ψ(F, f₀, s₀, Caᵢ, mp::BarbarottaRossiDedeQuarteroni2018)
# 	# Reproduction of https://onlinelibrary.wiley.com/doi/10.1002/cnm.3137
#     @unpack a, b, aᶠ, bᶠ, aˢ, bˢ, aᶠˢ, bᶠˢ, K, k = mp

# 	# Gram-Schmidt step to enforce strict orthogonality
# 	s₀ = s₀ - (f₀⋅s₀)*f₀
# 	# Reconstruct normal
# 	n₀ = cross(f₀, s₀)
# 	n₀ /= norm(n₀)
	
# 	# Transverse isotropic active contraction (incompressible) - Also Rossi?
# 	λ = λᵃ(Caᵢ)
# 	Fᵃ = λ*f₀⊗f₀ + (one(F) - f₀⊗f₀)/sqrt(λ)

# 	# Orthotropic active contraction (incompressible) - Rossi
# 	# ξᶠ = (λᵃ(Caᵢ)-1.0) # microscopic shortening
# 	# γf = ξᶠ
# 	# γn = ξᶠ
# 	# γs = 1.0/((1.0+γf)*(1.0+γn)) - 1.0
# 	# Fᵃ = one(F) + γf * f₀⊗f₀ + γs * s₀⊗s₀ + γn * n₀⊗n₀

# 	# Everything is computed in Fᵉ
# 	Fᵉ = F⋅inv(Fᵃ)
# 	C = tdot(Fᵉ)
# 	I₃ = det(C)
# 	J = sqrt(I₃)
# 	#J = det(Fᵉ)
#     I₁ = tr(C/cbrt(J^2))
# 	I₄ᶠ = f₀ ⋅ C ⋅ f₀
# 	I₄ˢ = s₀ ⋅ C ⋅ s₀
# 	I₈ᶠˢ = (f₀ ⋅ C ⋅ s₀ + s₀ ⋅ C ⋅ f₀)/2

# 	U = K/4.0 * (J^2 - 1 - 2*log(J))

# 	Ψ = a/(2.0*b)*exp(b*(I₁-3.0)) + aᶠˢ/(2.0*bᶠˢ)*(exp(bᶠˢ*I₈ᶠˢ^2)-1.0) + U
# 	if I₄ᶠ > 1.0
# 		Ψ += aᶠ/(2.0*bᶠ)*(exp(bᶠ*(I₄ᶠ - 1)^2)-1.0)
# 	end
# 	if I₄ˢ > 1.0
# 		Ψ += aˢ/(2.0*bˢ)*(exp(bˢ*(I₄ˢ - 1)^2)-1.0)
# 	end

#     return Ψ
# end

# """
# """
# struct Passive2017Energy #<: IsotropicMaterialModel
# 	a
# 	a₁
# 	a₂
# 	b
# 	α₁
# 	α₂
# 	β
# 	η
# end

# """
# """
# function Ψ(F, f₀, s₀, Caᵢ, mp::Passive2017Energy)
# 	# Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
#     @unpack a, a₁, a₂, b, α₁, α₂, β, η = mp
# 	C = tdot(F)
#     I₁ = tr(C)
# 	I₃ = det(C)
# 	I₄ = tr(C ⋅ f₀ ⊗ f₀)

# 	I₃ᵇ = I₃^b
# 	U = β * (I₃ᵇ + 1/I₃ᵇ - 2)^a
# 	# Ψᵖ = α₁*(I₁/cbrt(I₃) - 3)^a₁ + α₂*max((I₄ - 1), 0.0)^a₂ + U
# 	Ψᵖ = α₁*(I₁/cbrt(I₃) - 3)^a₁ + α₂*(I₄ - 1)^2 + U

# 	M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
# 	Fᵃ = Tensors.unsafe_symmetric(one(F) + (λᵃ(Caᵢ) - 1.0) * M)
# 	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
# 	M̃ = f̃ ⊗ f̃

# 	Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀)
# 	FMF = Tensors.unsafe_symmetric(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))
# 	Iᵉ₄ = tr(FMF)
# 	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2

#     return Ψᵖ + Ψᵃ
# end



# """
# """
# Base.@kwdef struct BioNeoHooekean #<: IsotropicMaterialModel
# 	α = 1.0
# 	β = 100.0
# 	a = 1
# 	b = 2
# 	η = 10.0
# end

# """
# """
# function Ψ(F, f₀, s₀, Caᵢ, mp::BioNeoHooekean)
# 	# Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
#     @unpack a, b, α, β, η = mp
# 	C = tdot(F)
#     I₁ = tr(C)
# 	I₃ = det(C)
# 	n₀ = cross(f₀,s₀)
# 	#I₄ = f₀ ⋅ C ⋅ f₀

# 	I₃ᵇ = I₃^b
# 	#U = β * (I₃ᵇ + 1/I₃ᵇ - 2)^a
# 	U = β*log(I₃)^2
# 	Ψᵖ = α*(I₁/cbrt(I₃) - 3) + U + (n₀ ⋅ C ⋅ n₀ - 1)^2 

# 	#M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
# 	M = f₀ ⊗ f₀
# 	Fᵃ = one(F) + (λᵃ(Caᵢ) - 1.0) * M
# 	#Fᵃ = Tensors.unsafe_symmetric(λᵃ(Caᵢ) * M + (1.0/sqrt(λᵃ(Caᵢ)))*(one(F) - M))

# 	#Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀
# 	Fᵉ = F⋅inv(Fᵃ)
# 	Cᵉ = tdot(Fᵉ)
# 	#I₃ᵉ = det(Cᵉ)
# 	#I₃ᵉᵇ = I₃ᵉ^b
# 	#Uᵃ = 1.0 * (I₃ᵉᵇ + 1/I₃ᵉᵇ - 2)^a

# 	#Iᵉ₁ = tr(Cᵉ)
# 	Iᵉ₄ = f₀ ⋅ Cᵉ ⋅ f₀
# 	#Ψᵃ = η / 2 * (Iᵉ₄/cbrt(I₃ᵉ) - 1)^2 + 1.0*(Iᵉ₁/cbrt(I₃ᵉ)-3) + Uᵃ 
# 	#Ψᵃ = 1.00*(Iᵉ₁/cbrt(I₃ᵉ)-3) #+ Uᵃ 
# 	#Ψᵃ = η / 2 * (Iᵉ₄/cbrt(I₃ᵉ) - 1)^2 + Uᵃ
# 	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2

#     return Ψᵖ + Ψᵃ
# end

