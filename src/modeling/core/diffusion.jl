# @doc raw"""
#     BilinearDiffusionIntegrator{CoefficientType}

# Assembles the matrix associated to the bilinearform ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
# """
"""
    Represents the integrand of the bilinear form <ϕ,ψ> = -∫ D∇ϕ ⋅ ∇ψ dΩ .
"""
struct BilinearDiffusionIntegrator{CoefficientType} <: AbstractBilinearIntegrator
    D::CoefficientType
    # coordinate_system
end

struct BilinearDiffusionElementCache{IT <: BilinearDiffusionIntegrator, CV}
    integrator::IT
    cellvalues::CV
end

function assemble_element!(Kₑ, cell, element_cache::BilinearDiffusionElementCache, time)
    @unpack cellvalues = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

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

setup_element_cache(element_model::BilinearDiffusionIntegrator, qr, ip, ip_geo) = BilinearDiffusionElementCache(element_model, CellValues(qr, ip, ip_geo))
