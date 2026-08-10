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

function field_name_of_weak_boundary_condition(
    bc::AbstractWeakBoundaryCondition,
    sdh::SubDofHandler,
)
    return if hasfield(typeof(bc), :field_name) && bc.field_name != :auto
        bc.field_name
    else
        @assert length(sdh.field_names) == 1 "For problems with multiple fields please pass the field name into your weak boundary conditions via `field_name`"
        first(sdh.field_names)
    end
end

@doc raw"""
    RobinBC(α, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \alpha \bm{u} \quad \textbf{x} \in \partial \Omega_0,
```

See [`ViscousRobinBC`](@ref) for the dashpot which resists the *rate* instead of the displacement.
"""
struct RobinBC <: AbstractWeakBoundaryCondition
    α::Float64
    boundary_name::String
    field_name::Symbol
end
RobinBC(a, b) = RobinBC(a, b, :auto)

@doc raw"""
    NormalSpringBC(kₛ, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - k_s (\bm{u} \cdot \bm{n}_0) \bm{n}_0 \quad \textbf{x} \in \partial \Omega_0,
```

See [`NormalViscousSpringBC`](@ref) for the dashpot which resists the *rate* instead of the
displacement.
"""
struct NormalSpringBC <: AbstractWeakBoundaryCondition
    kₛ::Float64
    boundary_name::String
    field_name::Symbol
end
NormalSpringBC(a, b) = NormalSpringBC(a, b, :auto)

@doc raw"""
    BendingSpringBC(kᵇ, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \partial_F \frac{1}{2} k_b \left (cof(F) n_0 - n_0 \right) \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct BendingSpringBC <: AbstractWeakBoundaryCondition
    kᵇ::Float64
    boundary_name::String
    field_name::Symbol
end
BendingSpringBC(a, b) = BendingSpringBC(a, b, :auto)

@doc raw"""
    ConstantPressureBC(p::Real, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - p n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct ConstantPressureBC <: AbstractWeakBoundaryCondition
    p::Float64
    boundary_name::String
    field_name::Symbol
end
ConstantPressureBC(a, b) = ConstantPressureBC(a, b, :auto)

@doc raw"""
    PressureFieldBC(pressure_field, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - p(\bm{x}, t) J \bm{F}^{-\mathrm{T}} \bm{n}_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct PressureFieldBC{C} <: AbstractWeakBoundaryCondition
    pc::C
    boundary_name::String
    field_name::Symbol
end
PressureFieldBC(a, b) = PressureFieldBC(a, b, :auto)

@doc raw"""
Supertype of the weak boundary conditions which resist the *rate* of the displacement rather than the
displacement itself, i.e. the dashpots of the Robin family.

```math
\bm{P} \cdot \bm{n}_0 = - \bm{D}(\bm{n}_0) \cdot \bm{v} \quad \textbf{x} \in \partial \Omega_0,
```

The whole family is linear in the facet velocity ``\bm{v}``, so a subtype is fully described by its
positive semi-definite damping tensor [`damping_tensor`](@ref); it needs no assembly code of its own.

Unlike the springs these need a velocity, which only a time scheme can supply. They are therefore
assembled through the `gto1` protocol and are rejected by `HomotopyPathSolver`, which is load stepping
and has no rate to offer. See [`ViscousRobinBC`](@ref) and [`NormalViscousSpringBC`](@ref).
"""
abstract type AbstractViscousWeakBoundaryCondition <: AbstractWeakBoundaryCondition end

@doc raw"""
    ViscousRobinBC(η, boundary_name::String [, field_name::Symbol])

Dashpot resisting the full velocity, the rate analogue of [`RobinBC`](@ref).

```math
\bm{P} \cdot \bm{n}_0 = - \eta \bm{v} \quad \textbf{x} \in \partial \Omega_0,
```

``\eta`` is a viscosity per unit reference area. The velocity is the one the time scheme reconstructs
from the unknown displacement, so under backward Euler this is
``\bm{v} = (\bm{u}_n - \bm{u}_{n-1}) / \Delta t_n``, and under Newmark the corresponding
``\gamma / (\beta \Delta t)`` reconstruction.
"""
struct ViscousRobinBC <: AbstractViscousWeakBoundaryCondition
    η::Float64
    boundary_name::String
    field_name::Symbol
end
ViscousRobinBC(a, b) = ViscousRobinBC(a, b, :auto)

@doc raw"""
    NormalViscousSpringBC(cₛ, boundary_name::String [, field_name::Symbol])

Dashpot resisting the normal velocity only, the rate analogue of [`NormalSpringBC`](@ref).

```math
\bm{P} \cdot \bm{n}_0 = - c_s (\bm{v} \cdot \bm{n}_0) \bm{n}_0 \quad \textbf{x} \in \partial \Omega_0,
```

Tangential sliding is left free, which is what makes this the usual companion of
[`NormalSpringBC`](@ref) when modelling the pericardium: the sac resists the chamber pushing against
it, not the chamber sliding within it.
"""
struct NormalViscousSpringBC <: AbstractViscousWeakBoundaryCondition
    cₛ::Float64
    boundary_name::String
    field_name::Symbol
end
NormalViscousSpringBC(a, b) = NormalViscousSpringBC(a, b, :auto)

@doc raw"""
    damping_tensor(bc::AbstractViscousWeakBoundaryCondition, n₀::Vec)

The damping tensor ``\bm{D}`` of a dashpot boundary condition, evaluated at a quadrature point with
reference normal ``n_0``.

This is the entire constitutive content of [`AbstractViscousWeakBoundaryCondition`](@ref): the
traction is ``-\bm{D} \cdot \bm{v}`` and, because that is linear in the velocity, ``\bm{D}`` is also
the sensitivity the tangent needs. A new dashpot is one method here.
"""
function damping_tensor end

@inline damping_tensor(bc::ViscousRobinBC, ::Vec{dim, T}) where {dim, T} =
    bc.η * one(SymmetricTensor{2, dim, T})
@inline damping_tensor(bc::NormalViscousSpringBC, n₀::Vec) = bc.cₛ * symmetric(n₀ ⊗ n₀)

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
    p,
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
    p,
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
    p,
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
    p,
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
    p,
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
    p,
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
    p,
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
    p,
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
    p,
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
    dof_range,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    # @info qp, J, cofF ⋅ n₀
    neumann_term = p * J * cofF ⋅ n₀
    # neumann_term = p * n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[dof_range[i]] += neumann_term ⋅ δuᵢ * dΓ

        for j = 1:ndofs_facet
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[dof_range[i], dof_range[j]] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_facet_pressure_qp!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
    dof_range,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[dof_range])
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
            Kₑ[dof_range[i], dof_range[j]] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_facet_pressure_qp!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
    dof_range,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    neumann_term = p * J * cofF ⋅ n₀
    # neumann_term = p * n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[dof_range[i]] += neumann_term ⋅ δuᵢ * dΓ
    end
end


function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:PressureFieldBC},
    p,
)
    @unpack mp, fv = cache
    @unpack pc = mp
    t = get_time(p)

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        pressure = evaluate_coefficient(pc, cell, qp, t)
        assemble_facet_pressure_qp!(Kₑ, residualₑ, uₑ, pressure, qp, fv, cache.dof_range)
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:PressureFieldBC},
    p,
)
    @unpack mp, fv = cache
    @unpack pc = mp
    t = get_time(p)

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        pressure = evaluate_coefficient(pc, cell, qp, t)
        assemble_facet_pressure_qp!(Kₑ, uₑ, pressure, qp, fv, cache.dof_range)
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:PressureFieldBC},
    p,
)
    @unpack mp, fv = cache
    @unpack pc = mp
    t = get_time(p)

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        pressure = evaluate_coefficient(pc, cell, qp, t)
        assemble_facet_pressure_qp!(residualₑ, uₑ, pressure, qp, fv, cache.dof_range)
    end
end



function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    p,
)
    @unpack mp, fv = cache
    pressure = mp.p

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(Kₑ, residualₑ, uₑ, pressure, qp, fv, cache.dof_range)
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    p,
)
    @unpack mp, fv = cache
    pressure = mp.p

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(Kₑ, uₑ, pressure, qp, fv, cache.dof_range)
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    p,
)
    @unpack mp, fv = cache
    pressure = mp.p

    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(residualₑ, uₑ, pressure, qp, fv, cache.dof_range)
    end
end


# --- viscous (dashpot) boundary conditions ------------------------------------------------------
#
# Same data as `SimpleFacetCache`, different type -- and the type is the whole point, since it is what
# selects the assembly protocol. A dashpot is a function of `(u, v)` rather than of `(u, t)`, and a
# `SimpleFacetCache` is handed the bare time, from which no rate can be formed. The same split, for the
# same reason, distinguishes `QuasiStaticElementCache` from the condensed caches in
# `solid/elements.jl`.
#
# NOT a `FerriteOperators.AbstractGenericFirstOrderTimeSurfaceElementCache`, despite needing the rate.
# That supertype hands a facet `(uₑprev, t, Δt)`, from which the *only* velocity that can be
# reconstructed is the backward Euler difference quotient -- which is wrong under Newmark, where the
# reconstruction slope is `γ/(βΔt)` rather than `1/Δt`. The scheme-aware conversion happens instead in
# `facet_velocity` (`solid/elements.jl`), which hands this cache an `AffineVelocity`. Subtyping would
# additionally make our `assemble_facet!` methods ambiguous against that supertype's own.

struct ViscousFacetCache{MP, FV} <: AbstractSurfaceElementCache
    mp::MP
    fv::FV
    dof_range::UnitRange{Int}
end
function duplicate_for_device(device, cache::ViscousFacetCache)
    return ViscousFacetCache(cache.mp, duplicate_for_device(device, cache.fv), cache.dof_range)
end
@inline is_facet_in_cache(facet::FacetIndex, cell::CellCache, facet_cache::ViscousFacetCache) =
    facet ∈ getfacetset(cell.grid, getboundaryname(facet_cache))
@inline getboundaryname(facet_cache::ViscousFacetCache) = facet_cache.mp.boundary_name

function setup_boundary_cache(
    facet_model::AbstractViscousWeakBoundaryCondition,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
)
    field_name = field_name_of_weak_boundary_condition(facet_model, sdh)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    dof_range  = Ferrite.dof_range(sdh, field_name)
    return ViscousFacetCache(facet_model, FacetValues(qr, ip, ip_geo), dof_range)
end

# A dashpot asks the parameter object for the velocity reconstruction, exactly as a spring asks it for
# the time with `get_time`. `facet_velocity` returns an `AffineVelocity` (`solid/elements.jl`), which
# is deliberately *not* a timestep: the two coincide only under backward Euler, and reading `Δt` here
# instead is what would make the dashpot silently wrong under Newmark.
#
# The whole family is linear in the velocity, so the damping tensor is both the traction sensitivity
# and the tangent, and no automatic differentiation is needed. The tangent is independent of `uₑ`.
@inline function _viscous_facet_velocity(cache::ViscousFacetCache, qp, uₑ, velocity)
    u_q  = function_value(cache.fv, qp, @view uₑ[cache.dof_range])
    uᵥ_q = function_value(cache.fv, qp, @view velocity.uᵥ[cache.dof_range])
    return velocity.∂v∂u * (u_q - uᵥ_q)
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index::Int,
    cache::ViscousFacetCache,
    p,
)
    @unpack mp, fv = cache
    velocity = facet_velocity(p)

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        D  = damping_tensor(mp, getnormal(fv, qp))

        traction = D ⋅ _viscous_facet_velocity(cache, qp, uₑ, velocity)

        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[cache.dof_range[i]] += δuᵢ ⋅ traction * dΓ

            δuᵢD = δuᵢ ⋅ D # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += velocity.∂v∂u * (δuᵢD ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function assemble_facet!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    cell,
    local_facet_index::Int,
    cache::ViscousFacetCache,
    p,
)
    @unpack mp, fv = cache
    velocity = facet_velocity(p)

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        D  = damping_tensor(mp, getnormal(fv, qp))

        for i = 1:ndofs_facet
            δuᵢD = shape_value(fv, qp, i) ⋅ D # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                Kₑ[cache.dof_range[i], cache.dof_range[j]] += velocity.∂v∂u * (δuᵢD ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function assemble_facet!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    cell,
    local_facet_index::Int,
    cache::ViscousFacetCache,
    p,
)
    @unpack mp, fv = cache
    velocity = facet_velocity(p)

    reinit!(fv, cell, local_facet_index)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        D  = damping_tensor(mp, getnormal(fv, qp))

        traction = D ⋅ _viscous_facet_velocity(cache, qp, uₑ, velocity)

        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[cache.dof_range[i]] += δuᵢ ⋅ traction * dΓ
        end
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
    p,
)
    (; Δ, inner_cache, Kₑfd, uₑfd, residualₑfd, residualₑref) = cache

    # The incoming element matrix might be non-empty, so we need to start by storing the offset.
    Kₑfd .= Kₑ

    # The actual assembly is happening here
    assemble_facet!(Kₑ, residualₑ, uₑ, cell, local_facet_index, inner_cache, p)

    # Now we get a fresh reference state to pull the differences
    fill!(residualₑref, 0.0)
    assemble_facet!(residualₑref, uₑ, cell, local_facet_index, inner_cache, p)
    # Here we actually compute teh finite difference
    for i = 1:length(uₑfd)
        fill!(residualₑfd, 0.0)
        uₑfd    .= uₑ
        uₑfd[i] += Δ
        assemble_facet!(residualₑfd, uₑfd, cell, local_facet_index, inner_cache, p)
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
