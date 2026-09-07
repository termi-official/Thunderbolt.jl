"""
    ChamberVolumeCoupling(chamber_surface_setname, lumped_volume_symbol, lumped_pressure_symbol, pressure_symbol_3D)

Descriptor for which volume to couple with which variable for the constraint.

`chamber_surface_setname` has to name a *closed* surface, e.g. the `"LVChamberSurface"` of
[`generate_ideal_lv_mesh`](@ref): the chamber volume is the divergence-theorem integral over it, and
that is the enclosed volume only where the surface closes.
"""
struct ChamberVolumeCoupling
    chamber_surface_setname::String
    # Untyped on purpose. These hold either a plain `Symbol` (hand-written lumped models) or a
    # `ModelingToolkit.Num` (MTK-backed ones), and naming the MTK type here would make this struct
    # undefinable without ModelingToolkit — which it must not be, since the `Symbol` flavour is used
    # without MTK. They are read exactly three times, all in `create_chamber_tyings`/`semidiscretize`,
    # and the value that reaches assembly is narrowed to `Symbol` by
    # `RSAFDQ2022SingleChamberTying.displacement_symbol`, so leaving them untyped costs no runtime.
    lumped_volume_symbol::Any
    lumped_pressure_symbol::Any
    pressure_symbol_3D::Symbol
end

"""
Enforce the constraints that
  chamber volume 3D (solid model) = chamber volume 0D (lumped circuit)
via Lagrange multipliers, where a surface pressure integral is introduced over each chamber surface.
The 3D volume is [`Thunderbolt.volume_integral`](@ref) over the closed chamber surface each
[`ChamberVolumeCoupling`](@ref) names.

This approach has been proposed by [RegSalAfrFedDedQar:2022:cem](@citet).
"""
struct LumpedFluidSolidCoupler <: AbstractCoupler
    chamber_couplings::Vector{ChamberVolumeCoupling}
    displacement_symbol::Any # see the note on `ChamberVolumeCoupling`
end

@doc raw"""
    volume_integral(x, d, F, N)

One quadrature point of the chamber volume `V³ᴰ(u)`, in the reference configuration.

```math
V = -\frac{1}{3} \oint_{\partial \Omega} (\bm{x} + \bm{d}) \cdot \bm{n} \, \mathrm{d}a
  = -\frac{1}{3} \oint_{\partial \Omega_0} (\bm{x} + \bm{d}) \cdot \mathrm{det}(\bm{F}) \bm{F}^{-T} \bm{N} \, \mathrm{d}A
```

the divergence theorem on the deformed chamber surface, pulled back with Nanson's formula. The sign
is the traversal's: a chamber surface is swept from the cells bounding the chamber, so `n` points
into it and the enclosed volume comes out positive.

This is a volume only where the surface is closed -- an endocardium open at the valvular orifice
gives a number that depends on the origin and is not the cavity. Closing it is the mesh's job, see
the `with_valvular_plane` keyword of [`generate_ideal_lv_mesh`](@ref).
"""
volume_integral(x::Vec, d::Vec, F::Tensor, N::Vec) = -det(F) * (x + d) ⋅ (transpose(inv(F)) ⋅ N) / 3

"""
    Pressure3D0DVolumeCoupler(chamber_surface_name, displacement_symbol, pressure_symbol)

The 3D side of one chamber's 3D-0D tying, as a facet term of the structural model.

It contributes the chamber volume constraint
    ∫ V³ᴰ(u) ∂Ω - V⁰ᴰ(c)
where u are the unknowns in the 3D problem and c the unknowns in the 0D problem, and the pressure
contribution (i.e. variation w.r.t. p) for the term
    ∫ p n(u) δu ∂Ω
 [= ∫ p J(u) F(u)^-T n₀ δu ∂Ω₀]
where p is the unknown chamber pressure and u contains the unknown deformation field.

One object owns the whole coupling contract: it declares the pressure as an algebraic variable, the
endocardial facets as facet items, the pressure dof as the facet items' global-dof tail, and the
`- V⁰ᴰ` row as its algebraic item. The chamber surface is named rather than resolved, so the term is
constructible where the model is written down and the facetset is looked up at setup.
"""
struct Pressure3D0DVolumeCoupler
    chamber_surface_name::String
    displacement_symbol::Symbol
    pressure_symbol::Symbol
end

algebraic_variables(model::Pressure3D0DVolumeCoupler) = (model.pressure_symbol,)

@concrete struct Pressure3D0DVolumeCouplerCache <: AbstractSurfaceElementCache
    fv
    displacement_range
    pressure_index
    # Which chamber this cache serves. `pressure_index` cannot say so: it is a position in the
    # subdomain's augmented tail, and the subdomains differ in `ndofs_per_cell`.
    pressure_symbol
end

duplicate_for_device(device, cache::Pressure3D0DVolumeCouplerCache) =
    Pressure3D0DVolumeCouplerCache(
        duplicate_for_device(device, cache.fv),
        cache.displacement_range,
        cache.pressure_index,
        cache.pressure_symbol,
    )

# The chamber pressure belongs to no cell, so it enters the facet items' local system as their
# global-dof tail; the endocardial facets carrying the term are their own traversal rather than a
# per-facet membership test on the cell sweep. The declared set spans whatever subdomains the chamber
# surface touches, and each subdomain declares the part it owns. The declaration is the facet items'
# alone, so the subdomain's volumetric kernels keep the pure displacement system and the pressure's
# sparsity is the tying surface, not the mesh.
FerriteOperators.facet_item_global_dofs(model::Pressure3D0DVolumeCoupler, sdh::SubDofHandler) =
    algebraic_dofs(sdh.dh, model.pressure_symbol)

FerriteOperators.facet_items(model::Pressure3D0DVolumeCoupler, sdh::SubDofHandler) = filter(
    facet -> facet[1] ∈ sdh.cellset,
    getfacetset(get_grid(sdh.dh), model.chamber_surface_name),
)

"""
    setup_facet_item_cache(model::Pressure3D0DVolumeCoupler, qr, sdh, global_dof_range)

The coupling cache for one chamber on one subdomain, with `global_dof_range` naming where that
chamber's pressure sits in the augmented local system `[celldofs(cell); global dofs]`.

The range is a parameter because several chambers may share a subdomain's tail, and only the
integrator holding all of them knows their order.
"""
function FerriteOperators.setup_facet_item_cache(
    model::Pressure3D0DVolumeCoupler,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
    global_dof_range,
)
    ip     = Ferrite.getfieldinterpolation(sdh, model.displacement_symbol)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return Pressure3D0DVolumeCouplerCache(
        FacetValues(qr, ip, ip_geo),
        dof_range(sdh, model.displacement_symbol),
        only(global_dof_range),
        model.pressure_symbol,
    )
end

"""
    _chamber_volume_facet(fv, coords, dₑ) -> V

One facet's contribution to `∫_Γ V³ᴰ(u) dΓ`, over a `FacetValues` already reinitialized on it.

The single spelling of the volume integral's quadrature loop. Three consumers integrate through it:
the tying kernel's chamber row, the [`ChamberVolumeFunctional`](@ref) reduction, and the reference
volume [`compute_chamber_volume`](@ref) evaluates before any operator exists — so none of the three
can drift from the others.
"""
function _chamber_volume_facet(fv, coords, dₑ)
    V = zero(eltype(dₑ))
    for qp = 1:getnquadpoints(fv)
        ∇d = function_gradient(fv, qp, dₑ)
        F  = one(∇d) + ∇d
        d  = function_value(fv, qp, dₑ)
        x  = spatial_coordinate(fv, qp, coords)
        V  += volume_integral(x, d, F, getnormal(fv, qp)) * getdetJdV(fv, qp)
    end
    return V
end

"""
    ChamberBalanceCache()

Algebraic cache for the chamber rows `r[p] -= V⁰ᴰ`. One cache serves every chamber;
`args.item.index` (via [`query_cell_parameters`](@ref)) selects which chamber's solver-supplied
reference volume applies. The row is constant in `u` -- the integral half, `∫_Γ V³ᴰ(u) dΓ`, is the
facet item kernel's -- so its Jacobian block is left to the AD fallback.
"""
struct ChamberBalanceCache end

duplicate_for_device(device, cache::ChamberBalanceCache) = cache

# `V⁰ᴰ` is solver-supplied data, not element state: it arrives fresh through `p` on every sweep
# rather than through a mutable field the cache (and its per-worker duplicates) would hold a
# reference to.
FerriteOperators.query_cell_parameters(
    ::ChamberBalanceCache,
    item::FerriteOperators.AlgebraicItem,
    p,
) = p.V⁰ᴰ[item.index]

FerriteOperators.assemble_algebraic!(
    req::FerriteOperators.ResidualRequest,
    ::ChamberBalanceCache,
    args::FerriteOperators.AlgebraicArgs,
) = (req.r[1] -= args.p; nothing)

# The `- V⁰ᴰ` half of the chamber row lives on no cell, so it is an item of the algebraic family,
# owned by the same term that writes the `∫_Γ V³ᴰ(u) dΓ` half onto the facets.
FerriteOperators.algebraic_items(model::Pressure3D0DVolumeCoupler, dh::DofHandler) =
    [[only(algebraic_dofs(dh, model.pressure_symbol))]]

FerriteOperators.setup_algebraic_cache(::Pressure3D0DVolumeCoupler, ::DofHandler) =
    ChamberBalanceCache()

FerriteOperators.provides_analytic(
    ::Type{<:Pressure3D0DVolumeCouplerCache},
    ::Union{FerriteOperators.JacobianKind{:u}, FerriteOperators.JacobianResidualKind},
) = true

FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::Pressure3D0DVolumeCouplerCache,
    args::FerriteOperators.FacetArgs,
    local_facet_index::Int,
) = _assemble_3D0D_coupling_facet!(req, cache, args, local_facet_index)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::Pressure3D0DVolumeCouplerCache,
    args::FerriteOperators.FacetArgs,
    local_facet_index::Int,
) = _assemble_3D0D_coupling_facet!(req, cache, args, local_facet_index)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::Pressure3D0DVolumeCouplerCache,
    args::FerriteOperators.FacetArgs,
    local_facet_index::Int,
) = _assemble_3D0D_coupling_facet!(req, cache, args, local_facet_index)

# One body for the three requests: which buffers it fills is decided on the request type, so the
# branches fold away per kernel.
function _assemble_3D0D_coupling_facet!(
    req,
    element_cache::Pressure3D0DVolumeCouplerCache,
    args::FerriteOperators.FacetArgs,
    local_facet_index::Int,
)
    (; fv, displacement_range, pressure_index) = element_cache
    geometry_cache = args.cell

    reinit!(fv, geometry_cache, local_facet_index)

    # The chamber pressure is the tail of the augmented local system, `[celldofs(cell); pressure]`.
    uₑ = args.states.u
    pdof = pressure_index
    dₑ = @view uₑ[displacement_range]
    p = uₑ[pdof]
    coords = getcoordinates(geometry_cache)

    residual =
        req isa Union{FerriteOperators.ResidualRequest, FerriteOperators.JacobianResidualRequest}
    jacobian =
        req isa
        Union{FerriteOperators.JacobianRequest{:u}, FerriteOperators.JacobianResidualRequest}

    for qp in QuadratureIterator(fv)
        # Part 1: Surface pressure part
        ∂Ω₀ = getdetJdV(fv, qp)

        ∇d = function_gradient(fv, qp, dₑ)
        F = one(∇d) + ∇d
        J = det(F)
        invF = inv(F)
        cofF = transpose(invF)

        n₀ = getnormal(fv, qp)
        n = cofF ⋅ n₀

        for i ∈ 1:getnbasefunctions(fv)
            δuᵢ = shape_value(fv, qp, i)
            residual && (req.r[displacement_range[i]] += p * J * n ⋅ δuᵢ * ∂Ω₀)
            if jacobian
                for j ∈ 1:getnbasefunctions(fv)
                    ∇δuⱼ = shape_gradient(fv, qp, j)
                    # Add contribution to the tangent
                    #   δF^-1 = -F^-1 δF F^-1
                    #   δJ = J tr(δF F^-1)
                    # Product rule
                    δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
                    δJ = J * tr(∇δuⱼ ⋅ invF)
                    δJcofF = δJ * cofF + J * δcofF
                    req.K[displacement_range[i], displacement_range[j]] +=
                        p * (δJcofF ⋅ n₀) ⋅ δuᵢ * ∂Ω₀
                end
                req.K[displacement_range[i], pdof] += J * n ⋅ δuᵢ * ∂Ω₀
            end
        end

        # Part 2: Chamber volume constraint part. Its residual is the whole facet integral, taken
        # once below through `_chamber_volume_facet`; what is left here is the tangent.
        if jacobian
            d = function_value(fv, qp, dₑ)
            x = spatial_coordinate(fv, qp, coords)
            # Via chain rule we obtain:
            #   δV(u,F(u)) = δu ⋅ dVdu + δF : dVdF
            ∂V∂u = Tensors.gradient(u_ -> volume_integral(x, u_, F, n₀), d)
            ∂V∂F = Tensors.gradient(u_ -> volume_integral(x, d, u_, n₀), F)
            for j ∈ 1:getnbasefunctions(fv)
                δuⱼ = shape_value(fv, qp, j)
                ∇δuⱼ = shape_gradient(fv, qp, j)
                req.K[pdof, displacement_range[j]] += (∂V∂u ⋅ δuⱼ + ∂V∂F ⊡ ∇δuⱼ) * ∂Ω₀
            end
            # req.K[pdof, pdof] += 0
        end
    end

    residual && (req.r[pdof] += _chamber_volume_facet(fv, coords, dₑ))

    return nothing
end

"""
    ChamberVolumeFunctional(pressure_symbol)

The reduction `V³ᴰ = ∫_Γ V³ᴰ(u) dΓ` over one chamber's tying facets.

`pressure_symbol` names the chamber, matching the [`Pressure3D0DVolumeCoupler`](@ref) that declared
it. One sweep evaluates one chamber: a facet belonging to a different chamber contributes nothing,
so an operator carrying several couplers is swept once per chamber. Only the facet-item family can
carry a surface integral, so the kind is declared over that family alone — an operator without the
coupler's facet items fails the reduction's precondition instead of answering a silent zero.

Evaluate through [`chamber_volume`](@ref).
"""
struct ChamberVolumeFunctional
    pressure_symbol::Symbol
end

FerriteOperators.reduction_families(::Type{ChamberVolumeFunctional}) = (:facets,)
functional_value_type(::ChamberVolumeFunctional) = Float64

function FerriteOperators.evaluate_facet_functional(
    kind::ChamberVolumeFunctional,
    cache::Pressure3D0DVolumeCouplerCache,
    args::FerriteOperators.FacetArgs,
    local_facet_index::Int,
)
    kind.pressure_symbol === cache.pressure_symbol || return nothing
    reinit!(cache.fv, args.cell, local_facet_index)
    dₑ = @view args.states.u[cache.displacement_range]
    return _chamber_volume_facet(cache.fv, getcoordinates(args.cell), dₑ)
end

# The chambers an operator's tying facets serve. A reduction for a symbol none of them names would
# come back as a silent zero -- every facet declining is a legitimate empty sum to the engine -- so
# `chamber_volume` rejects it here instead.
_chamber_symbols(cache) = ()
_chamber_symbols(cache::Pressure3D0DVolumeCouplerCache) = (cache.pressure_symbol,)
_chamber_symbols(cache::FerriteOperators.CompositeFacetItemCache) =
    Tuple(sym for inner in cache.inner_caches for sym in _chamber_symbols(inner))
_tied_chamber_symbols(op) = unique!(
    Symbol[
        sym for sc in get_subdomain_caches(op) if sc.domain isa FacetItemDomain for
        sym in _chamber_symbols(sc.domain.element)
    ],
)

"""
    chamber_volume(op, pressure_symbol, states, p = nothing, ctx = nothing) -> V³ᴰ

The 3D chamber volume `∫_Γ V³ᴰ(u) dΓ` of the chamber named by `pressure_symbol`, reduced over the
tying facets `op` assembles.

`states` carries the operator's slots (`(u = u,)` for a stationary evaluation). Nothing is written
into `op`, and the integrand is the chamber row's own, so this reports the volume the coupled
residual balances against `V⁰ᴰ`. A symbol naming no chamber of `op` is an `ArgumentError`.
"""
function chamber_volume(op, pressure_symbol::Symbol, states::NamedTuple, p = nothing, ctx = nothing)
    served = _tied_chamber_symbols(op)
    pressure_symbol ∈ served || throw(
        ArgumentError(
            "The operator carries no tying facets for a chamber named `$pressure_symbol`; it serves " *
            "$(served). A chamber volume is the integral over that chamber's own declared facets.",
        ),
    )
    return FerriteOperators.evaluate_functional(
        op,
        ChamberVolumeFunctional(pressure_symbol),
        states,
        p,
        ctx,
    )
end
chamber_volume(op, pressure_symbol::Symbol, u::AbstractVector, p = nothing, ctx = nothing) =
    chamber_volume(op, pressure_symbol, (u = u,), p, ctx)
