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
    reinit!(element_cache.cellvalues, cell)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        ξ = cellvalues.qr.points[q_point]
        qp = QuadraturePoint(q_point, ξ)
        ρ = evaluate_coefficient(element_cache.integrator.ρ, cell, qp, time)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, q_point, j)
                Mₑ[i,j] += ρ * Nᵢ * Nⱼ * dΩ 
            end
        end
    end
end
