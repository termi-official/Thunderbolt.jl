# @doc raw"""
#     AssembledDiffusionOperator{MT, DT, CV}

# Assembles the matrix associated to the bilinearform ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
# """
"""
    Represents the integrand of the bilinear form <ϕ,ψ> = -∫ D∇ϕ ⋅ ∇ψ dΩ .
"""
struct BilinearDiffusionIntegrator{CoefficientType}
    D::CoefficientType
    # coordinate_system
end

struct BilinearDiffusionElementCache{IT <: BilinearDiffusionIntegrator, CV}
    integrator::IT
    cellvalues::CV
end

function assemble_element!(Kₑ, cell, element_cache::CACHE, time) where {CACHE <: BilinearDiffusionElementCache}
    @unpack cellvalues = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    Ferrite.reinit!(cellvalues, cell)

    for qp in QuadratureIterator(cellvalues)
        D_loc = evaluate_coefficient(element_cache.integrator.D, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i,j] -= ((D_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
            end
        end
    end
end
