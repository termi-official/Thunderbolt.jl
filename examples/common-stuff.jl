using Thunderbolt, UnPack, SparseArrays
import Thunderbolt: Ψ, U

# struct MicrostructureCache{sdim}
#     reference_fibers::Vector{Vec{sdim}}
#     reference_sheets::Vector{Vec{sdim}}
#     reference_normals::Vector{Vec{sdim}}
# end

# function directions(cache::MicrostructureCache{dim}, qp::Int) where {dim}
#     reference_fibers[qp], reference_sheets[qp], reference_normals[qp]
# end

# struct ElementIntegralForm{M}
#     model::M
#     subdomain::String
#     cvs::CellValueCollection
# end

# struct BoundaryIntegralForm{M}
#     model::M
#     subdomain::String
#     fvs::FaceValueCollection
# end

# struct DiscreteSteadyStateProblem
#     element_integrals
#     face_integrals
# end


using UnPack

Base.@kwdef struct NewActiveSpring{CPT}
    a::Float64   = 2.7
    aᶠ::Float64  = 1.6
    mpU::CPT = NullCompressionPenalty()
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring{CPT}) where {CPT}
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
    I₃ = det(C)
    J = det(F)
    I₁ = tr(C/cbrt(J^2))
    I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0)^2 + Thunderbolt.U(I₃, mpU)
end


Base.@kwdef struct SimpleActiveSpring
    aᶠ::Float64  = 1.0
end
using UnPack
function Thunderbolt.Ψ(F, Fᵃ, f₀, s₀, n₀, mp::SimpleActiveSpring)
    @unpack aᶠ = mp

    Fᵉ = F ⋅ inv(Fᵃ)
    Cᵉ = tdot(Fᵉ)
    Iᵉ₄ᶠ = f₀ ⋅ Cᵉ ⋅ f₀

    return aᶠ/2.0*(Iᵉ₄ᶠ-1.0)^2
end

# """
# Our reported fit against LinYin.
# """
Base.@kwdef struct NewActiveSpring2{CPT}
    # a::Float64   = 15.020986456784657
    # aᶠ::Float64  =  4.562365556553194
    a::Float64   = 2.6
    aᶠ::Float64  =  1.6
    mpU::CPT = NullCompressionPenalty()
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring2{CPT}) where {CPT}
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
    I₃ = det(C)
    J = det(F)
    I₁ = tr(C/cbrt(J^2))
    I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0) + Thunderbolt.U(I₃, mpU)
end

Base.@kwdef struct NewHolzapfelOgden2009Model{TD,TU} #<: OrthotropicMaterialModel
    a₁::TD  =  5.0
    a₂::TD  =  2.5
    a₃::TD  =  0.5
    a₄::TD  = 10.0
    b₁::TD  =  5.0
    b₂::TD  =  2.5
    b₃::TD  =  0.5
    b₄::TD  =  2.0
    mpU::TU = SimpleCompressionPenalty(50.0)
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewHolzapfelOgden2009Model)
    @unpack a₁, b₁, a₂, b₂, a₃, b₃, a₄, b₄, mpU = mp

    C = tdot(F)
    I₃ = det(C)
    J = det(F)

    I₄ᶠ = f₀ ⋅ C ⋅ f₀ / cbrt(J^2)
    I₄ˢ = s₀ ⋅ C ⋅ s₀ / cbrt(J^2)
    I₄ⁿ = n₀ ⋅ C ⋅ n₀ / cbrt(J^2)
    I₈ᶠˢ = (f₀ ⋅ C ⋅ s₀ + s₀ ⋅ C ⋅ f₀)/2.0

    Ψᵖ = a₁/(2.0*b₁)*exp(b₁*(I₄ᶠ - 1.0)) + a₂/(2.0*b₂)*exp(b₂*(I₄ˢ -1.0)) + a₃/(2.0*b₃)*exp(b₃*(I₄ⁿ - 1.0)) + a₄/(2.0*b₄)*exp(b₄*(I₈ᶠˢ - 1.0)) + U(I₃, mpU)
    return Ψᵖ
end

"""
In 'A transmurally heterogeneous orthotropic activation model for ventricular contraction and its numerical validation' it is suggested that uniform activtaion is fine.

TODO citation.
"""
struct CalciumHatField end

"""
"""
Thunderbolt.evaluate_coefficient(coeff::CalciumHatField, cell_cache, qp, t) = t < 1.0 ? t : 2.0-t
