"""
"""
abstract type AbstractIonChannel end;

"""
"""
abstract type MarkovTypeIonChannel <: AbstractIonChannel end;

"""
Helper to properly dispatch individual gating variables.
"""
abstract type HodgkinHuxleyTypeGate end;

"""
Parameters for a generic sigmoid function of the form

σ(x) = \frac{A + B x}{C + D \exp{\frac{E + F x}{G}}}
"""
struct GenericSigmoidParameters{T}
    A::T
    B::T
    C::T
    D::T
    E::T
    F::T
    G::T
end

@inline σ(s::T1, p::GenericSigmoidParameters{T2}) where {T1, T2} = (p.A + p.B*s)/(p.C + p.D*exp(p.E + p.F*s) / p.G)

"""
The classical gate formulation is stated in the normalized affine form:

∂ₜ𝐬ₖ = αₖ(φₘ)𝐬ₖ + βₖ(φₘ)

where αₖ(φₘ) = \frac{A + B φₘ}{C + D \exp{\frac{E + F φₘ}{G}}}

Note that the original formulation is

∂ₜ𝐬ₖ = aₖ(φₘ)𝐬ₖ + bₖ(φₘ)(1 - 𝐬ₖ)

where αₖ = aₖ - bₖ and βₖ = bₖ.
"""
struct GenericHodgkinHuxleyGate{T} <: HodgkinHuxleyTypeGate where {T}
    αₚ::GenericSigmoidParameters{T}
    βₚ::GenericSigmoidParameters{T}
end

@inline α(g::GenericHodgkinHuxleyGate{T}, φₘ::T) where {T} = σ(φₘ, g.αₚ)
@inline β(g::GenericHodgkinHuxleyGate{T}, φₘ::T) where {T} = σ(φₘ, g.βₚ)

# Spatially varying parameters
@inline α(g::GenericHodgkinHuxleyGate{T1}, φₘ::T2, x::T3) where {T1,T2,T3} = σ(φₘ, g.αₚ(x))
@inline β(g::GenericHodgkinHuxleyGate{T1}, φₘ::T2, x::T3) where {T1,T2,T3} = σ(φₘ, g.βₚ(x))

"""
Probabilistic ion channels with diagonal, semi-affine internal structure.

∂ₜ𝐬₁ = g₁(φₘ, 𝐬) = α₁(φₘ)𝐬₁ + β₁(φₘ)
        .
        .
        .
∂ₜ𝐬ₙ = gₙ(φₘ, 𝐬) = αₙ(φₘ)𝐬ₙ + βₙ(φₘ)

They can be derived as special cases of Markov type ion channels with
tensor-product structure (TODO citation). 𝐬 is called the gating vector
and its entries are the gating variables.
"""
struct HodgkinHuxleyTypeIonChannel{NGates} <: AbstractIonChannel where {NGates}
    gates::SVector{NGates, HodgkinHuxleyTypeGate}
    powers::SVector{NGates, Int}
end;

@inline function g(gate::HodgkinHuxleyTypeGate, φₘ::T, 𝐬ᵢ::T) where {T}
    α(gate, φₘ)*𝐬ᵢ + β(gate, φₘ)
end

@inline function g(gate::HodgkinHuxleyTypeGate, φₘ::T, 𝐬ᵢ::T, x::AbstractVector{T}) where {T}
    α(gate, φₘ, x)*𝐬ᵢ + β(gate, φₘ, x)
end

"""
Ohmic current of the form

Iⱼ = ̄gⱼ pⱼ (φₘ - Eⱼ)

where ̄gⱼ is the maximal conductance, pᵢ the open probability of the associated channel and Eⱼ the equilibrium potential.
"""
struct OhmicCurrent{T, NChannels}
    g::T
    channels::SVector{NChannels, HodgkinHuxleyTypeIonChannel}
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
struct HodgkinHuxleyModel <: HodgkinHuxleyTypeModel end;

"""
Simplest model with qubic reaction and no state.

(TODO citation)
"""
struct NagumoModel <: HodgkinHuxleyTypeModel end;

"""
Simple model with qubic reaction and linear state.

(TODO citation)
"""
struct FitzHughNagumoModel <: HodgkinHuxleyTypeModel end;

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
    stim::AbstractStimulationProtocol
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
    stim::AbstractStimulationProtocol
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
