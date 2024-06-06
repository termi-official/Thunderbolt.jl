
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

#TODO Energy-based interface
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

struct PressureFieldBC{C}
    pc::C
    boundary_name::String
end



struct EmtpyFacetCache
end

assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index, face_caches::EmtpyFacetCache, time) = nothing
assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, face_caches::EmtpyFacetCache, time) = nothing



function setup_boundary_cache(boundary_models::Union{<:Tuple,<:AbstractVector}, qr::FacetQuadratureRule, ip, ip_geo)
    length(boundary_models) == 0 && return EmtpyFacetCache()
    return ntuple(i->setup_boundary_cache(boundary_models[i], qr, ip, ip_geo), length(boundary_models))
end

function assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, face_caches::Tuple, time)
    for face_cache ∈ face_caches
        if (cellid(cell), local_face_index) ∈ getfacetset(cell.grid, getboundaryname(face_cache))
            assemble_face!(Kₑ, uₑ, cell, local_face_index, face_cache, time)
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index, face_caches::Tuple, time)
    for face_cache ∈ face_caches
        if (cellid(cell), local_face_index) ∈ getfacetset(cell.grid, getboundaryname(face_cache))
            assemble_face!(Kₑ, residualₑ, uₑ, cell, local_face_index, face_cache, time)
        end
    end
end



struct SimpleFacetCache{MP, FV}
    mp::MP
    fv::FV
end

getboundaryname(face_cache::SimpleFacetCache) = face_cache.mp.boundary_name

function setup_boundary_cache(face_model, qr::FacetQuadratureRule, ip, ip_geo)
    return SimpleFacetCache(face_model, FacetValues(qr, ip, ip_geo))
end


function assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index::Int, cache::SimpleFacetCache{<:RobinBC}, time)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
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

function assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:RobinBC}, time)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
    
            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end



function assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:NormalSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
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

function assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:NormalSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)
    
        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)
    
        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
    
            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end



function assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:BendingSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            ∇δui = shape_gradient(fv, qp, i)
            residualₑ[i] += ∇δui ⊡ ∂Ψ∂F * dΓ

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j in 1:ndofs_face
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
            end
        end
    end
end

function assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:BendingSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)
    
        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u
    
        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)
    
        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            ∇δui = shape_gradient(fv, qp, i)
    
            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j in 1:ndofs_face
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
            end
        end
    end
end



function assemble_face_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv::FacetValues)
    ndofs_face = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    neumann_term = p * J * cofF
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ

        for j in 1:ndofs_face
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[i, j] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_face_pressure_qp!(Kₑ, uₑ, p, qp, fv::FacetValues)
    ndofs_face = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u
    
    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    # neumann_term = p * J * cofF
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)

        for j in 1:ndofs_face
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[i, j] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:PressureFieldBC}, time)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_face_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv)
    end
end

function assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:PressureFieldBC}, time)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        # Add contribution to the residual from this test function
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_face_pressure_qp!(Kₑ, uₑ, p, qp, fv)
    end
end



function assemble_face!(Kₑ::Matrix, residualₑ, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:ConstantPressureBC}, time)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv)
    end
end


function assemble_face!(Kₑ::Matrix, uₑ, cell, local_face_index, cache::SimpleFacetCache{<:ConstantPressureBC}, time)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(Kₑ, uₑ, p, qp, fv)
    end
end
