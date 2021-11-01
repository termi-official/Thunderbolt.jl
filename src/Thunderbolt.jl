module Thunderbolt

abstract type AbstractIonChannel end;

abstract type MarkovTypeIonChannel <: AbstractIonChannel end;

"""
Helper to properly dispatch individual gating variables.
"""
abstract type HodgkinHuxleyTypeGate end;

"""
The classical gate formulation has the form

τₖ(φₘ)∂ₜsₖ = σ(φₘ) - φₘ
"""
struct ClassicalHodgkinHuxleyGate <: HodgkinHuxleyTypeGate

end

"""
Ion channels with diagonal, semi-affine internal structure.

∂ₜ𝐬₁ = g₁(φₘ, 𝐬) = α₁(φₘ)𝐬₁ + β₁(φₘ)
     .
     .
     .
∂ₜ𝐬ₙ = gₙ(φₘ, 𝐬) = αₙ(φₘ)𝐬ₙ + βₙ(φₘ)

They can be derived as special cases of Markov type ion channels with
tensor-product structure (TODO citation). 𝐬 is called the gating vector
and its entries are the gating variables.
"""
struct HodgkinHuxleyTypeIonChannel <: AbstractIonChannel
    gates::Vector{HodgkinHuxleyTypeGate}
end;

@inline function g(::HodgkinHuxleyTypeGate, φₘ::T, 𝐬ᵢ::T) where {T}
    α(::HodgkinHuxleyTypeGate, φₘ)*𝐬 + β(::HodgkinHuxleyTypeGate, φₘ)
end

@inline function g(::HodgkinHuxleyTypeGate, φₘ::T, 𝐬ᵢ::T, x::AbstractVector{T}) where {T}
    α(::HodgkinHuxleyTypeGate, φₘ, x)*𝐬 + β(::HodgkinHuxleyTypeGate, φₘ, x)
end

abstract type AbstractIonicModel end;

"""
Models where all states are described by Hodgkin-Huxley type ion channels.
"""
abstract type HodgkinHuxleyTypeModel <: AbstractIonicModel end;

"""
The model from the seminal paper of Hodgkin and Huxley (1952).

(TODO citation)
"""
struct HodgkinHuxleyModel end <: HodgkinHuxleyTypeModel;

"""
Simplest model with qubic reaction and no state.

(TODO citation)
"""
struct NagumoModel end <: HodgkinHuxleyTypeModel;

"""
Simple model with qubic reaction and linear state.

(TODO citation)
"""
struct FitzHughNagumoModel end <: HodgkinHuxleyTypeModel;


abstract type AbstractEPModel end;

abstract type AbstractStimulationProtocol end;

"""
Assumtion: Iₛₜᵢₘ,ₑ = Iₛₜᵢₘ,ᵢ.
"""
abstract type TransmembraneStimulationProtocol <: AbstractStimulationProtocol end;

"""
The original model formulation (TODO citation) with the structure

 χCₘ∂ₜφₘ = ∇⋅κᵢ∇φᵢ + χ(Iᵢₒₙ(φₘ,𝐬,x) + Iₛₜᵢₘ,ᵢ(x,t))
 χCₘ∂ₜφₘ = ∇⋅κₑ∇φₑ - χ(Iᵢₒₙ(φₘ,𝐬,x) + Iₛₜᵢₘ,ₑ(x,t))
    ∂ₜ𝐬  = g(φₘ,𝐬,x)
 φᵢ - φₑ = φₘ

"""
struct ParabolicParabolicBidomainModel <: AbstractEPModel
    χ
    Cₘ
    κᵢ
    κₑ
    stim::AbstractStimulationProtocoll
    ion::AbstractIonicModel
end

"""
Transformed bidomain model with the structure

 χCₘ∂ₜφₘ = ∇⋅κᵢ∇φₘ + ∇⋅κᵢ∇φₑ      + χ(Iᵢₒₙ(φₘ,𝐬,x) + Iₛₜᵢₘ(x,t))
      0  = ∇⋅κᵢ∇φₘ + ∇⋅(κᵢ+κₑ)∇φₑ +  Iₛₜᵢₘ,ₑ(t) - Iₛₜᵢₘ,ᵢ(t)
    ∂ₜ𝐬  = g(φₘ,𝐬,x)
      φᵢ = φₘ + φₑ

This formulation is a transformation of the parabolic-parabolic
form (c.f. TODO citation) and has been derived by (TODO citation) first.
"""
struct ParabolicEllipticBidomainModel <: AbstractEPModel
    χ
    Cₘ
    κᵢ
    κₑ
    stim::AbstractStimulationProtocoll
    ion::AbstractIonicModel
end


"""
Simplification of the bidomain model with the structure

 χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ(φₘ,𝐬) + Iₛₜᵢₘ(t))
    ∂ₜ𝐬  = g(φₘ,𝐬)

(TODO citation). Can be derived through the assumption (TODO), but also when the
assumption is violated we can construct optimal κ (TODO citation+example) for the
reconstruction of φₘ.
"""
struct MonodomainModel <: AbstractEPModel
    χ
    Cₘ
    κ
    stim::TransmembraneStimulationProtocol
    ion::AbstractIonicModel
end

end