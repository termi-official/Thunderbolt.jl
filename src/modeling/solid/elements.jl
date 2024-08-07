"""
    StructuralElementCache

A generic cache to assemble elements coming from a [StructuralModel](@ref).
"""
struct StructuralElementCache{M, CCache, CMCache, CV} <: AbstractVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_model_cache::CMCache
    cv::CV
end

# TODO how to control dispatch on required input for the material routin?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, geometry_cache::CellCache, element_cache::StructuralElementCache, time)
    @unpack constitutive_model, internal_model_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        internal_state = state(internal_model_cache, geometry_cache, qp, time)
        P, ∂P∂F = material_routine(constitutive_model, coefficient_cache, F, internal_state, geometry_cache, qp, time)

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

function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, geometry_cache::CellCache, element_cache::StructuralElementCache, time)
    @unpack constitutive_model, internal_model_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        internal_state = state(internal_model_cache, geometry_cache, qp, time)
        P, ∂P∂F = material_routine(constitutive_model, coefficient_cache, F, internal_state, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            # residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, geometry_cache::CellCache, element_cache::StructuralElementCache, time)
    @unpack constitutive_model, internal_model_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        internal_state = state(internal_model_cache, geometry_cache, qp, time)
        P, ∂P∂F = material_routine(constitutive_model, coefficient_cache, F, internal_state, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            # ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            # for j in 1:ndofs
            #     ∇δuj = shape_gradient(cv, qp, j)
            #     # Add contribution to the tangent
            #     Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            # end
        end
    end
end

function setup_element_cache(model::QuasiStaticModel, qr::QuadratureRule, ip, sdh)
    ip_geo = geometric_subdomain_interpolation(sdh)
    cv = CellValues(qr, ip, ip_geo)
    return StructuralElementCache(
        model,
        setup_coefficient_cache(model, qr, sdh),
        setup_internal_model_cache(model, qr, sdh),
        cv
    )
end
