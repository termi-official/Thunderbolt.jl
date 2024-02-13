# @doc raw"""
#     AssembledMassOperator{MT, CV}

# Assembles the matrix associated to the bilinearform ``a(u,v) = -\int v(x) u(x) dx`` for ``u,v`` from the same function space.
# """
"""
    Represents the integrand of the bilinear form <ϕ,ψ> = ∫ ρϕ ⋅ ψ dΩ .
"""
struct BilinearMassIntegrator{CoefficientType}
    ρ::CoefficientType
    # coordinate_system
end

struct BilinearMassElementCache{IT <: BilinearMassIntegrator, T, CV}
    integrator::IT
    ρq::Vector{T}
    cellvalues::CV
end

function assemble_element!(Mₑ, cell, element_cache::CACHE, time) where {CACHE <: BilinearMassElementCache}
    @unpack cellvalues = element_cache
    Ferrite.reinit!(element_cache.cellvalues, cell)
    n_basefuncs = getnbasefunctions(cellvalues)
    for qp in QuadratureIterator(cellvalues)
        ρ = evaluate_coefficient(element_cache.integrator.ρ, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, qp, i)
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, qp, j)
                Mₑ[i,j] += ρ * Nᵢ * Nⱼ * dΩ 
            end
        end
    end
end
