function setup_boundary_cache(boundary_models::Tuple, qr::FacetQuadratureRule, sdh::SubDofHandler)
    length(boundary_models) == 0 && return EmptySurfaceElementCache()
    return CompositeSurfaceElementCache(
        ntuple(i->setup_boundary_cache(boundary_models[i], qr, sdh), length(boundary_models)),
    )
end

@doc raw"""
Any boundary condition stated in the weak form
```math
\int f(u, n_0), \delta u \mathrm{d} \partial \Omega_0
```
"""
abstract type AbstractWeakBoundaryCondition end

function field_name_of_weak_boundary_condition(bc::AbstractWeakBoundaryCondition, sdh::SubDofHandler)
    return if hasfield(typeof(bc), :field_name) && bc.field_name != :auto
        bc.field_name
    else 
        @assert length(sdh.field_names) == 1 "For problems with multiple fields please pass the field name into your weak boundary conditions via `field_name`"
        first(sdh.field_names)
    end
end

@doc raw"""
    RobinBC(α, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \alpha \bm{u} \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct RobinBC <: AbstractWeakBoundaryCondition
    α::Float64
    boundary_name::String
    field_name::Symbol
end
RobinBC(a,b) = RobinBC(a, b, :auto)

@doc raw"""
    NormalSpringBC(kₛ boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - k_s \bm{u} \cdot n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct NormalSpringBC <: AbstractWeakBoundaryCondition
    kₛ::Float64
    boundary_name::String
    field_name::Symbol
end
NormalSpringBC(a,b) = NormalSpringBC(a, b, :auto)

@doc raw"""
    BendingSpringBC(kᵇ, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \partial_F \frac{1}{2} k_b \left (cof(F) n_0 - n_0 \right) \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct BendingSpringBC <: AbstractWeakBoundaryCondition
    kᵇ::Float64
    boundary_name::String
    field_name::Symbol
end
BendingSpringBC(a,b) = BendingSpringBC(a, b, :auto)

@doc raw"""
    ConstantPressureBC(p::Real, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - p n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct ConstantPressureBC <: AbstractWeakBoundaryCondition
    p::Float64
    boundary_name::String
    field_name::Symbol
end
ConstantPressureBC(a,b) = ConstantPressureBC(a, b, :auto)

@doc raw"""
    PressureFieldBC(pressure_field, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - k_s \bm{u} \cdot n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct PressureFieldBC{C} <: AbstractWeakBoundaryCondition
    pc::C
    boundary_name::String
    field_name::Symbol
end
PressureFieldBC(a,b) = PressureFieldBC(a, b, :auto)

"""
Standard cache for surface integrals.
"""
struct SimpleFacetCache{MP, FV} <: AbstractSurfaceElementCache
    mp::MP
    fv::FV
    dof_range::UnitRange{Int}
end
function duplicate_for_device(device, cache::SimpleFacetCache)
    return SimpleFacetCache(cache.mp, duplicate_for_device(device, cache.fv), cache.dof_range)
end
@inline is_facet_in_cache(facet::FacetIndex, cell::CellCache, facet_cache::SimpleFacetCache) =
    facet ∈ getfacetset(cell.grid, getboundaryname(facet_cache))
@inline getboundaryname(facet_cache::SimpleFacetCache) = facet_cache.mp.boundary_name

function setup_boundary_cache(
    facet_model::AbstractWeakBoundaryCondition,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
)
    field_name = field_name_of_weak_boundary_condition(facet_model, sdh)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    dof_range  = Ferrite.dof_range(sdh, field_name)
    return SimpleFacetCache(facet_model, FacetValues(qr, ip, ip_geo), dof_range)
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index::Int,
    cache::SimpleFacetCache{<:RobinBC},
    time,
)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += (δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:RobinBC},
    time,
)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)

            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += (δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index::Int,
    cache::SimpleFacetCache{<:RobinBC},
    time,
)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂Ψ∂u = Tensors.gradient(u -> α*u⋅u, u_q)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ
        end
    end
end



function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:NormalSpringBC},
    time,
)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += (δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:NormalSpringBC},
    time,
)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)

            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += (δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:NormalSpringBC},
    time,
)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂Ψ∂u = Tensors.gradient(u -> 0.5*kₛ*(u⋅N)^2, u_q)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ
        end
    end
end



function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:BendingSpringBC},
    time,
)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
            F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N),
            F,
            :all,
        )

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            ∇δui = shape_gradient(fv, qp, i)
            residualₑ[cache.dof_range[i]] += ∇δui ⊡ ∂Ψ∂F * dΓ

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j = 1:ndofs_facet
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += (∇δui∂P∂F ⊡ ∇δuj) * dΓ
            end
        end
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:BendingSpringBC},
    time,
)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
            F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N),
            F,
            :all,
        )

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            ∇δui = shape_gradient(fv, qp, i)

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j = 1:ndofs_facet
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += (∇δui∂P∂F ⊡ ∇δuj) * dΓ
            end
        end
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:BendingSpringBC},
    time,
)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
        F = one(∇u) + ∇u

        ∂Ψ∂F =
            Tensors.gradient(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            ∇δui = shape_gradient(fv, qp, i)
            residualₑ[cache.dof_range[i]] += ∇δui ⊡ ∂Ψ∂F * dΓ
        end
    end
end



function assemble_facet_pressure_qp!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    # @info qp, J, cofF ⋅ n₀
    neumann_term = p * J * cofF ⋅ n₀
    # neumann_term = p * n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[cache.dof_range[i]] += neumann_term ⋅ δuᵢ * dΓ

        for j = 1:ndofs_facet
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[cache.dof_range[i], cache.dof_range[j]] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_facet_pressure_qp!(Kₑ::AbstractMatrix, uₑ::AbstractVector, p, qp, fv::FacetValues)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    # neumann_term = p * J * cofF ⋅ n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)

        for j = 1:ndofs_facet
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[cache.dof_range[i], cache.dof_range[j]] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_facet_pressure_qp!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    neumann_term = p * J * cofF ⋅ n₀
    # neumann_term = p * n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[cache.dof_range[i]] += neumann_term ⋅ δuᵢ * dΓ
    end
end


function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:PressureFieldBC},
    time,
)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_facet_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv)
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:PressureFieldBC},
    time,
)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        # Add contribution to the residual from this test function
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_facet_pressure_qp!(Kₑ, uₑ, p, qp, fv)
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:PressureFieldBC},
    time,
)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_facet_pressure_qp!(residualₑ, uₑ, p, qp, fv)
    end
end



function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    time,
)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv)
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    time,
)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(Kₑ, uₑ, p, qp, fv)
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    time,
)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(residualₑ, uₑ, p, qp, fv)
    end
end


# We can use this to debug weak BCs for their consistency
struct ConsistencyCheckWeakBoundaryCondition{BC} <: AbstractWeakBoundaryCondition
    bc::BC
    Δ::Float64
end

struct ConsistencyCheckWeakBoundaryConditionCache{IC} <: AbstractSurfaceElementCache
    inner_cache::IC
    Kₑfd::Matrix{Float64}
    uₑfd::Vector{Float64}
    residualₑfd::Vector{Float64}
    residualₑref::Vector{Float64}
    Δ::Float64
end
function duplicate_for_device(device, cache::ConsistencyCheckWeakBoundaryConditionCache)
    return ConsistencyCheckWeakBoundaryConditionCache(
        duplicate_for_device(cache.inner_cache),
        duplicate_for_device(cache.Kₑfd),
        duplicate_for_device(cache.uₑfd),
        duplicate_for_device(cache.residualₑfd),
        duplicate_for_device(cache.residualₑref),
        cache.Δ,
    )
end
@inline is_facet_in_cache(
    facet::FacetIndex,
    cell::CellCache,
    facet_cache::ConsistencyCheckWeakBoundaryConditionCache,
) = is_facet_in_cache(facet, cell, facet_cache.inner_cache)
@inline getboundaryname(facet_cache::ConsistencyCheckWeakBoundaryConditionCache) =
    getboundaryname(facet_cache.inner_cache)
@inline getboundaryname(check::ConsistencyCheckWeakBoundaryCondition) = getboundaryname(check.bc)

function setup_boundary_cache(
    ccc::ConsistencyCheckWeakBoundaryCondition,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
)
    N = ndofs_per_cell(sdh)
    return ConsistencyCheckWeakBoundaryConditionCache(
        setup_boundary_cache(ccc.bc, qr, sdh),
        zeros(N, N),
        zeros(N),
        zeros(N),
        zeros(N),
        ccc.Δ,
    )
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index::Int,
    cache::ConsistencyCheckWeakBoundaryConditionCache,
    time,
)
    (; Δ, inner_cache, Kₑfd, uₑfd, residualₑfd, residualₑref) = cache

    # The incoming element matrix might be non-empty, so we need to start by storing the offset.
    Kₑfd .= Kₑ

    # The actual assembly is happening here
    assemble_facet!(Kₑ, residualₑ, uₑ, cell, local_facet_index, inner_cache, time)

    # Now we get a fresh reference state to pull the differences
    fill!(residualₑref, 0.0)
    assemble_facet!(residualₑref, uₑ, cell, local_facet_index, inner_cache, time)
    # Here we actually compute teh finite difference
    for i = 1:length(uₑfd)
        fill!(residualₑfd, 0.0)
        uₑfd    .= uₑ
        uₑfd[i] += Δ
        assemble_facet!(residualₑfd, uₑfd, cell, local_facet_index, inner_cache, time)
        residualₑfd .-= residualₑref
        residualₑfd /= Δ
        Kₑfd[:, i] .+= residualₑfd
    end

    # Finally we check for consistency
    if maximum(abs.(Kₑfd .- Kₑ)) > Δ
        @warn "Inconsistent element $(cellid(cell)) facet $(local_facet_index)! Jacobian difference: $(maximum(abs.(Kₑfd .- Kₑ)))"
        @info uₑ
    end
end
