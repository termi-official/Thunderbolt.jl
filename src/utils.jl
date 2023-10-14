function calculate_element_volume(cell, cellvalues_u, uₑ)
    reinit!(cellvalues_u, cell)
    evol::Float64=0.0;
    @inbounds for qp in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        ∇u = function_gradient(cellvalues_u, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        evol += J * dΩ
    end
    return evol
end;

function calculate_volume_deformed_mesh(w, dh::DofHandler, cellvalues_u)
    evol::Float64 = 0.0;
    @inbounds for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        nu = getnbasefunctions(cellvalues_u)
        global_dofs_u = global_dofs[1:nu]
        uₑ = w[global_dofs_u]
        δevol = calculate_element_volume(cell, cellvalues_u, uₑ)
        evol += δevol;
    end
    return evol
end;

@inline angle(v1::Vec{dim,T}, v2::Vec{dim,T}) where {dim, T} = acos((v1 ⋅ v2)/(norm(v1)*norm(v2)))
@inline angle_deg(v1::Vec{dim,T}, v2::Vec{dim,T}) where {dim, T} =  rad2deg(angle(v1, v2))

"""
    unproject(v::Vec{dim,T}, n::Vec{dim,T}, α::T)::Vec{dim, T}

Unproject the vector `v` from the plane with normal `n` such that the angle between `v` and the
resulting vector is `α` (given in radians).

!!! note It is assumed that the vectors are normalized and orthogonal, i.e. `||v|| = 1`, `||n|| = 1`
         and `v \\cdot n = 0`.
"""
@inline function unproject(v::Vec{dim,T}, n::Vec{dim,T}, α::T)::Vec{dim, T} where {dim, T}
    @debug @assert norm(v) ≈ 1.0
    @debug @assert norm(n) ≈ 1.0
    @debug @assert v ⋅ n ≈ 0.0

    α ≈ π/2.0 && return n # special case to prevent division by 0

    λ = (sqrt(1-cos(α)^2))/cos(α)
    return v + λ * n
end

"""
    rotate_around(v::Vec{dim,T}, a::Vec{dim,T}, θ::T)::Vec{dim,T}

Perform a Rodrigues' rotation of the vector `v` around the axis `a` with `θ` radians.

!!! note It is assumed that the vectors are normalized, i.e. `||v|| = 1` and `||a|| = 1`.
"""
@inline function rotate_around(v::Vec{dim,T}, a::Vec{dim,T}, θ::T)::Vec{dim,T} where {dim, T}
    @debug @assert norm(n) ≈ 1.0

    return v * cos(θ) + (a × v) * sin(θ) + a * (a ⋅ v) * (1-cos(θ))
end

"""
    orthogonalize(v₁::Vec{dim,T}, v₂::Vec{dim,T})::Vec{dim,T}

Returns a new `v₁` which is orthogonal to `v₂`.
"""
@inline function orthogonalize(v₁::Vec{dim,T}, v₂::Vec{dim,T})::Vec{dim,T} where {dim, T} 
    return v₁ - (v₁ ⋅ v₂)*v₂
end

"""
"""
function generate_nodal_quadrature_rule(ip::Interpolation{ref_shape, order}) where {ref_shape, order}
    n_base = Ferrite.getnbasefunctions(ip)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape, Float64}(ones(length(positions)), positions)
end
