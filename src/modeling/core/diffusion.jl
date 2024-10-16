@doc raw"""
    BilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct BilinearDiffusionIntegrator{CoefficientType} <: AbstractBilinearIntegrator
    D::CoefficientType
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
struct BilinearDiffusionElementCache{CoefficientCacheType, CV} <: AbstractVolumetricElementCache
    Dcache::CoefficientCacheType
    cellvalues::CV
end

function assemble_element!(Kₑ::AbstractMatrix, cell, element_cache::BilinearDiffusionElementCache, time)
    @unpack cellvalues, Dcache = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in QuadratureIterator(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i,j] -= _inner_product_helper(∇Nⱼ, D_loc, ∇Nᵢ) * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::BilinearDiffusionIntegrator, qr, ip, sdh::SubDofHandler)
    ip_geo = geometric_subdomain_interpolation(sdh)
    BilinearDiffusionElementCache(setup_coefficient_cache(element_model.D, qr, sdh), CellValues(qr, ip, ip_geo))
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble interface diffusion matrices.
"""
struct BilinearDiffusionInterfaceCache{CoefficientCacheType, IV} <: AbstractInterfaceElementCache
    Dcache_here::CoefficientCacheType
    Dcache_there::CoefficientCacheType
    interfacevalues::IV
end

function assemble_interface!(Ki::AbstractMatrix, interface_cache, interface_diffusion_cache::BilinearDiffusionInterfaceCache, time, γ = 1000.0)
    @unpack interfacevalues, Dcache_here, Dcache_there = interface_diffusion_cache
    reinit!(interfacevalues, interface_cache)
    h = 5
    for qp in QuadratureIterator(interfacevalues.here)
        D_here = evaluate_coefficient(Dcache_here, interface_cache, qp, time)
        D_there = evaluate_coefficient(Dcache_there, interface_cache, qp, time)
        normal = getnormal(interfacevalues, qp.i)
        dΓ = getdetJdV(interfacevalues, qp.i)
        for i in 1:getnbasefunctions(interfacevalues)
            is_here = i <= getnbasefunctions(interfacevalues.here)
            D = is_here ? D_here : D_there 
            μ = γ * normal ⋅ (D ⋅ normal)/h
            μ < 0 && @info μ
            ∇δu = shape_gradient(interfacevalues, qp.i, i; here = is_here)
            δu_jump = shape_value_jump(interfacevalues, qp.i, i) * (-normal)
            D∇δu_avg = (D ⋅ ∇δu)/2
            for j in 1:getnbasefunctions(interfacevalues)
                δis_here = j <= getnbasefunctions(interfacevalues.here)
                δD = δis_here ? D_here : D_there  
                u_jump = shape_value_jump(interfacevalues, qp.i, j) * (-normal)
                ∇u = shape_gradient(interfacevalues, qp.i, j;  here = δis_here)
                D∇u_avg = (δD ⋅ ∇u)/2
                # Add contribution to Ki
                Ki[i, j] += (δu_jump ⋅ D∇u_avg + D∇δu_avg ⋅ u_jump)*dΓ -  μ * (δu_jump ⋅ u_jump) * dΓ
            end
        end
    end
end

function setup_interface_cache(element_model::BilinearDiffusionIntegrator, qr, ip, sdh::SubDofHandler)
    ip_geo = geometric_subdomain_interpolation(sdh)
    D_here = setup_coefficient_cache(element_model.D, qr, sdh)
    D_there = setup_coefficient_cache(element_model.D, qr, sdh)
    BilinearDiffusionInterfaceCache(D_here, D_there, InterfaceValues(qr, ip, ip_geo))
end

@doc raw"""
    TransientDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\partial_t u = \nabla \cdot \kappa(x) \nabla u + f``
"""
struct TransientDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end

@doc raw"""
    SteadyDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\nabla \cdot \kappa(x) \nabla u = f``
"""
struct SteadyDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end
