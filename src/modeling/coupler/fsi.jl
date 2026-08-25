"""
Descriptor for which volume to couple with which variable for the constraint.
"""
struct ChamberVolumeCoupling{CVM}
    chamber_surface_setname::String
    control_point_setname::String
    chamber_volume_method::CVM
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
via Lagrange multiplied, where a surface pressure integral is introduced such that
  ∫  ∂Ωendo
Here `chamber_volume_method` is responsible to compute the 3D volume.

This approach has been proposed by [RegSalAfrFedDedQar:2022:cem](@citet).
"""
struct LumpedFluidSolidCoupler{CVM} <: AbstractCoupler
    chamber_couplings::Vector{ChamberVolumeCoupling{CVM}}
    displacement_symbol::Any # see the note on `ChamberVolumeCoupling`
end

"""
    Debug helper for FSI. Just keeps the chamber volume constant.
"""
struct ConstantChamberVolume
    volume::Float64
end

function volume_integral(x, d, F, N, method::ConstantChamberVolume)
    method.volume
end

"""
Chamber volume estimator as presented in [HirBasJagWilGee:2017:mcc](@cite).

Compute the chamber volume as a surface integral via the integral
   - ∫ (x + d) det(F) cof(F) N ∂Ωendo
where it is assumed that the chamber is convex, zero displacement in
apicobasal direction at the valvular plane occurs and the plane normal is aligned
with the z axis, where the origin is at z=0.
"""
struct Hirschvogel2017SurrogateVolume end

function volume_integral(x::Vec, d::Vec, F::Tensor, N::Vec, method::Hirschvogel2017SurrogateVolume)
    val = det(F) * (x + d) ⋅ inv(transpose(F)) ⋅ N
    return -val
end

"""
Chamber volume contribution for the 3D-0D constraint
    ∫ V³ᴰ(u) ∂Ω - V⁰ᴰ(c)
where u are the unkowns in the 3D problem and c the unkowns in the 0D problem.

Pressure contribution (i.e. variation w.r.t. p) for the term
    ∫ p n(u) δu ∂Ω
 [= ∫ p J(u) F(u)^-T n₀ δu ∂Ω₀]
where p is the unknown chamber pressure and u contains the unknown deformation field.
"""
struct Pressure3D0DVolumeCouplerIntegrator <: AbstractNonlinearIntegrator
    fqrc::FacetQuadratureRuleCollection
    displacement_symbol::Symbol
    pressure_symbol::Symbol
    facets::OrderedSet
    volume_method
end

@concrete struct Pressure3D0DVolumeCouplerCache <: AbstractSurfaceElementCache
    fv
    displacement_range
    pressure_index
    volume_method
end

duplicate_for_device(device, cache::Pressure3D0DVolumeCouplerCache) =
    Pressure3D0DVolumeCouplerCache(
        duplicate_for_device(device, cache.fv),
        cache.displacement_range,
        cache.pressure_index,
        cache.volume_method,
    )

# The chamber pressure belongs to no cell, so it enters the element-local system as its global-dof
# tail; the endocardial facets carrying the term are their own traversal rather than a per-facet
# membership test on the cell sweep.
FerriteOperators.global_dofs(model::Pressure3D0DVolumeCouplerIntegrator, sdh::SubDofHandler) =
    algebraic_dofs(sdh.dh, model.pressure_symbol)

FerriteOperators.facet_items(model::Pressure3D0DVolumeCouplerIntegrator, sdh::SubDofHandler) =
    filter(facet -> facet[1] ∈ sdh.cellset, model.facets)

FerriteOperators.setup_facet_item_cache(
    model::Pressure3D0DVolumeCouplerIntegrator,
    sdh::SubDofHandler,
) = setup_3D0D_coupling_cache(
    model,
    sdh,
    first(FerriteOperators.global_dof_range(model, sdh)),
)

"""
    setup_3D0D_coupling_cache(model, sdh, pressure_index)

The coupling cache for one chamber on one subdomain, with `pressure_index` naming where that
chamber's pressure sits in the augmented local system `[celldofs(cell); global dofs]`.

The index is a parameter because several chambers may share a subdomain's tail, and only the caller
holding all of them knows their order.
"""
function setup_3D0D_coupling_cache(
    model::Pressure3D0DVolumeCouplerIntegrator,
    sdh::SubDofHandler,
    pressure_index::Int,
)
    qr     = getquadraturerule(model.fqrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, model.displacement_symbol)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return Pressure3D0DVolumeCouplerCache(
        FacetValues(qr, ip, ip_geo),
        dof_range(sdh, model.displacement_symbol),
        pressure_index,
        model.volume_method,
    )
end

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
    (; fv, displacement_range, pressure_index, volume_method) = element_cache
    geometry_cache = args.cell

    reinit!(fv, geometry_cache, local_facet_index)

    # The chamber pressure is the tail of the augmented local system, `[celldofs(cell); pressure]`.
    uₑ = args.states.u
    pdof = pressure_index
    dₑ = @view uₑ[displacement_range]
    p = uₑ[pdof]
    coords = getcoordinates(geometry_cache)

    residual = req isa Union{
        FerriteOperators.ResidualRequest,
        FerriteOperators.JacobianResidualRequest,
    }
    jacobian = req isa Union{
        FerriteOperators.JacobianRequest{:u},
        FerriteOperators.JacobianResidualRequest,
    }

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

        # Part 2: Chamber volume constraint part
        d = function_value(fv, qp, dₑ)
        x = spatial_coordinate(fv, qp, coords)

        residual && (req.r[pdof] += volume_integral(x, d, F, n₀, volume_method) * ∂Ω₀)

        if jacobian
            # Via chain rule we obtain:
            #   δV(u,F(u)) = δu ⋅ dVdu + δF : dVdF
            ∂V∂u = Tensors.gradient(u_ -> volume_integral(x, u_, F, n₀, volume_method), d)
            ∂V∂F = Tensors.gradient(u_ -> volume_integral(x, d, u_, n₀, volume_method), F)
            for j ∈ 1:getnbasefunctions(fv)
                δuⱼ = shape_value(fv, qp, j)
                ∇δuⱼ = shape_gradient(fv, qp, j)
                req.K[pdof, displacement_range[j]] += (∂V∂u ⋅ δuⱼ + ∂V∂F ⊡ ∇δuⱼ) * ∂Ω₀
            end
            # req.K[pdof, pdof] += 0
        end
    end

    return nothing
end
