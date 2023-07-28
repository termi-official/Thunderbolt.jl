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

# struct RitzGalerkinSpatialDiscretization <: SpatialDiscretization
#     interpolation_collections::Dict{Symbol,InterpolationCollection}
#     qrs::Dict{Symbol,QuadratureCollection}
#     grid::Grid
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

# function discretize(problem, discretization)
#     ???
# end

# struct DiscreteSteadyStateProblem
#     element_integrals
#     face_integrals
# end

# ----------------------------------------------

mutable struct LazyMicrostructureCache{MM, VT}
    const microstructure_model::MM
    const x_ref::Vector{VT}
    cellid::Int
end

function Thunderbolt.directions(cache::LazyMicrostructureCache{MM}, qp::Int) where {MM}
    return directions(cache.microstructure_model, cache.cellid, cache.x_ref[qp])
end

function setup_microstructure_cache(cv, model::OrthotropicMicrostructureModel{FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}) where {FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}
    return LazyMicrostructureCache(model, cv.qr.points, -1)
end

function update_microstructure_cache!(cache::LazyMicrostructureCache{MM}, time::Float64, cell::CellCacheType, cv::CV) where {CellCacheType, CV, MM}
    cache.cellid = cellid(cell)
end


# ----------------------------------------------


struct PelceSunLangeveld1995Cache
    calcium_values::Vector{Float64}
end

function state(cache::PelceSunLangeveld1995Cache, qp::Int)
    return cache.calcium_values[qp]
end

function setup_contraction_model_cache(cv::CV, contraction_model::PelceSunLangeveld1995Model, cf::CF) where {CV, CF}
    return PelceSunLangeveld1995Cache(Vector{Float64}(undef, getnquadpoints(cv)))
end

function update_contraction_model_cache!(cache::PelceSunLangeveld1995Cache, time::Float64, cell::CellCacheType, cv::CV, calcium_field::CF) where {CellCacheType, CV, CF}
    for qp ∈ 1:getnquadpoints(cv)
        x_ref = cv.qr.points[qp]
        cache.calcium_values[qp] = value(calcium_field, Ferrite.cellid(cell), x_ref, time)
    end
end


# ----------------------------------------------


struct CardiacMechanicalElementCache{MP, MSCache, CMCache, CV}
    mp::MP
    microstructure_cache::MSCache
    # coordinate_system_cache::CSCache
    contraction_model_cache::CMCache
    cv::CV
end

function update_element_cache!(cache::CardiacMechanicalElementCache{MP, MSCache, CMCache, CV}, calcium_field::CF, time::Float64, cell::CellCacheType) where {CellCacheType, MP, MSCache, CMCache, CV, CF}
    update_microstructure_cache!(cache.microstructure_cache, time, cell, cache.cv)
    update_contraction_model_cache!(cache.contraction_model_cache, time, cell, cache.cv, calcium_field)
end

function assemble_element!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, vₑ::Vector, cache::CardiacMechanicalElementCache)
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


# ----------------------------------------------

@doc raw"""
    RobinBC

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \alpha \bm{u} \quad \textbf{x} \in \partial \Omega,
```
"""
struct RobinBC
    α::Float64
    boundary_name::String
end

#TODO Energy-based interface?
struct NormalSpringBC
    kₛ::Float64
    boundary_name::String
end

struct BendingSpringBC
    kᵇ::Float64
    boundary_name::String
end

struct ConstantPressureBC
    p::Float64
    boundary_name::String
end

struct SimpleFaceCache{MP, FV}
    mp::MP
    # time::Float64
    # microstructure_model::MM
    # coordinate_system::CS
    fv::FV
end

getboundaryname(face_cache::FC) where {FC} = face_cache.mp.boundary_name

function setup_face_cache(bcd::BCD, fv::FV) where {BCD, FV}
    SimpleFaceCache(bcd, fv)
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{RobinBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack α = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{NormalSpringBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack kₛ = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{BendingSpringBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            ∇δui = shape_gradient(fv, qp, i)
            residualₑ[i] -= ∇δui ⊡ ∂Ψ∂F * dΓ

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j in 1:ndofs_face
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
            end
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{ConstantPressureBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack p = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)

        n₀ = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        # ∂P∂F = Tensors.gradient(
        #     F_ad -> p * det(F_ad) * transpose(inv(F_ad)),
        # F)

        # Add contribution to the residual from this test function
        cofF = transpose(inv(F))
        neumann_term = p * det(F) * cofF
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ

            # ∂P∂Fδui =   ∂P∂F ⊡ (n₀ ⊗ δuᵢ) # Hoisted computation
            for j in 1:ndofs_face
                ∇δuⱼ = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                # Kₑ[i, j] += (n₀ ⊗ δuⱼ) ⊡ ∂P∂Fδui * dΓ
                Kₑ[i, j] += δuᵢ ⋅ (((cofF ⊡ ∇δuⱼ) * one(cofF) - cofF ⋅ transpose(∇δuⱼ)) ⋅ neumann_term) ⋅ n₀ * dΓ
            end
        end
    end
end

function update_face_cache(cell::CC, face_cache::SimpleFaceCache{MP}) where {CC, MP}
end

# ----------------------------------------------


"""
TODO rewrite on per field basis.
"""
function assemble_global!(K::SparseMatrixCSC, f::Vector, dh::DH, cv::CV, fv::FV, material_model::MM, uₜ::Vector, uₜ₋₁::Vector, microstructure_model::MSM, calcium_field::CF, t::Float64, Δt::Float64, face_models::FM) where {DH, CV, FV, MM, MSM, CF, FM}
    n = ndofs_per_cell(dh)
    Kₑ = zeros(n, n)
    residualₑ = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    microstructure_cache = setup_microstructure_cache(cv, microstructure_model)
    # TODO this should be outside of this routine, because the contraction might be stateful! But where/how? Maybe all caches should be factored out...?
    contraction_cache = setup_contraction_model_cache(cv, material_model.contraction_model, calcium_field)
    element_cache = CardiacMechanicalElementCache(material_model, microstructure_cache, contraction_cache, cv)

    face_caches = ntuple(i->setup_face_cache(face_models[i], fv), length(face_models))

    # Loop over all cells in the grid
    # @timeit "assemble" for cell in CellIterator(dh)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)

        # TODO refactor
        uₑ = uₜ[global_dofs] # element dofs
        uₑ_prev = uₜ₋₁[global_dofs] # element dofs
        vₑ = (uₑ - uₑ_prev)/Δt # velocity approximation

        # Reinitialize cell values, and reset output arrays
        reinit!(cv, cell)
        fill!(Kₑ, 0.0)
        fill!(residualₑ, 0.0)

        # Update remaining caches specific to the element and models
        update_element_cache!(element_cache, calcium_field, t, cell)

        # Assemble matrix and residuals
        assemble_element!(Kₑ, residualₑ, uₑ, vₑ, element_cache)

        for local_face_index ∈ 1:nfaces(cell)
            face_is_initialized = false
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    if !face_is_initialized
                        face_is_initialized = true
                        reinit!(fv, cell, local_face_index)
                    end
                    update_face_cache(cell, face_cache)

                    assemble_face!(Kₑ, residualₑ, uₑ, face_cache)
                end
            end
        end

        assemble!(assembler, global_dofs, residualₑ, Kₑ)
    end
end


# ----------------------------------------------


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
"""
struct CalciumHatField end

"""
"""
value(coeff::CalciumHatField, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim} = t < 1.0 ? t : 2.0-t

"""
Parameterization from Vallespin paper.

TODO citations
"""
Base.@kwdef struct Guccione1991Passive{CPT}
    C₀::Float64  =   0.1
    Bᶠᶠ::Float64 =  29.8
    Bˢˢ::Float64 =  14.9
    Bⁿⁿ::Float64 =  14.9
    Bⁿˢ::Float64 =   9.3
    Bᶠˢ::Float64 =  19.2
    Bᶠⁿ::Float64 =  14.4
    mpU::CPT = SimpleCompressionPenalty(50.0)
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::Guccione1991Passive{CPT}) where {CPT}
    @unpack C₀, Bᶠᶠ, Bˢˢ, Bⁿⁿ, Bⁿˢ, Bᶠˢ, Bᶠⁿ, mpU = mp

    C  = tdot(F)
    I₃ = det(C)
    J  = det(F)

    E  = (C-one(F))/2.0

    Eᶠᶠ = f₀ ⋅ E ⋅ f₀
    Eˢˢ = s₀ ⋅ E ⋅ s₀
    Eⁿⁿ = n₀ ⋅ E ⋅ n₀

    Eᶠˢ = f₀ ⋅ E ⋅ s₀
    Eˢᶠ = s₀ ⋅ E ⋅ f₀ 

    Eˢⁿ = s₀ ⋅ E ⋅ n₀
    Eⁿˢ = n₀ ⋅ E ⋅ s₀

    Eᶠⁿ = f₀ ⋅ E ⋅ n₀
    Eⁿᶠ = n₀ ⋅ E ⋅ f₀

    Q = Bᶠᶠ*Eᶠᶠ^2 + Bˢˢ*Eˢˢ^2 + Bⁿⁿ*Eⁿⁿ^2 + Bⁿˢ*(Eⁿˢ^2+Eˢⁿ^2) + Bᶠˢ*(Eᶠˢ^2+Eˢᶠ^2) + Bᶠⁿ*(Eᶠⁿ^2+Eⁿᶠ^2)

    return C₀*exp(Q)/2.0 + Thunderbolt.U(I₃, mpU)
end

Base.@kwdef struct Guccione1993Active
    Tₘₐₓ::Float64 = 100.0
end

Thunderbolt.∂(sas::Guccione1993Active, Caᵢ, F::Tensor{2, dim}, f₀::Vec{dim}, s₀::Vec{dim}, n₀::Vec{dim}) where {dim} = sas.Tₘₐₓ * Caᵢ * (F ⋅ f₀ ) ⊗ f₀
