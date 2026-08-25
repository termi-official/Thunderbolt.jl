# The quasi-static element caches, one per problem class. All three hold the same data; they differ
# only in the kinematics their local problem consumes, which is what selects the slots they read and
# the request kinds they serve:
#
# | cache                                  | local problem per quadrature point | slots read              |
# | :------------------------------------- | :--------------------------------- | :---------------------- |
# | `QuasiStaticElementCache`              | none, or algebraic `L(F, Q) = 0`   | `u`                     |
# | `QuasiStaticCondensedODEElementCache`  | `dₜQ = L(F, Q)`                    | `u`, `q`, `qprev`       |
# | `QuasiStaticCondensedDAEElementCache`  | `dₜQ = L(F, dₜF, Q)`               | `u`, `q`, `qprev`, `v`  |
#
# The DAE cache is the only one whose residual depends on the deformation rate, so it is the only one
# whose tangent carries the scheme's rate chain rule — see `assemble_cell!(::WeightedJacobianRequest, …)`.
#
# Julia's single inheritance means these cannot also share a Thunderbolt abstract parent, hence the
# `AnyQuasiStaticElementCache` union below for the methods that genuinely do not care.

"""
    QuasiStaticElementCache

A generic cache to assemble elements coming from a [StructuralModel](@ref).

Right now the model has to be formulated in the first Piola Kirchhoff stress tensor and F.

This is the **rate-free** variant: its local problem, if it has one at all, carries no time derivative,
so it needs neither a timestep nor a previous state. It is what continuation solvers such as
`HomotopyPathSolver` use.
"""
struct QuasiStaticElementCache{M, CCache, CMCache, CV} <:
       FerriteOperators.AbstractVolumetricElementCache
    # This one determines the exact material
    constitutive_model::M
    # This one is a helper to evaluate coefficients in a type stable way without allocations
    coefficient_cache::CCache
    # This one is a helper to condense local variables
    internal_cache::CMCache
    # FEValue scratch for the ansatz space
    cv::CV
end

"""
    QuasiStaticCondensedODEElementCache

Quasi-static element whose internal variable follows `dₜQ = L(F, Q)`. Together with the (singular)
mass matrix of the internal variables this is an ODE in mass matrix form. Its local problem reads the
deformation gradient alone, so its tangent is a plain `∂F/∂u`.
"""
struct QuasiStaticCondensedODEElementCache{M, CCache, CMCache, CV} <:
       FerriteOperators.AbstractVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
end

"""
    QuasiStaticCondensedDAEElementCache

Quasi-static element whose internal variable follows `dₜQ = L(F, dₜF, Q)`. The dependence on the rate
of the deformation gradient makes this a genuine DAE rather than a mass matrix ODE, and it is what
makes the `v` slot and the weighted tangent below part of this cache's contract.
"""
struct QuasiStaticCondensedDAEElementCache{M, CCache, CMCache, CV} <:
       FerriteOperators.AbstractVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
end

# Methods that only touch the shared fields dispatch on this union.
const AnyQuasiStaticElementCache = Union{
    QuasiStaticElementCache,
    QuasiStaticCondensedODEElementCache,
    QuasiStaticCondensedDAEElementCache,
}

# The two caches whose internal variable is condensed out of a *time dependent* local problem, i.e.
# those that get `q`, `qprev` and a timestep.
const QuasiStaticCondensedElementCache =
    Union{QuasiStaticCondensedODEElementCache, QuasiStaticCondensedDAEElementCache}

FerriteOperators.reinit_values!(e::AnyQuasiStaticElementCache, cell) = reinit!(e.cv, cell)
Ferrite.getnquadpoints(e::AnyQuasiStaticElementCache) = getnquadpoints(e.cv)

# The condensed state is written by `condense_cell!` and gathered back as the `q` slot; declaring this
# is what makes the sensitivity admissibility rules apply to these caches.
FerriteOperators.has_internal_state(::Type{<:QuasiStaticCondensedODEElementCache}) = true
FerriteOperators.has_internal_state(::Type{<:QuasiStaticCondensedDAEElementCache}) = true

"""
    get_number_of_internal_dofs_per_element(integrator, element_cache, sdh)

Number of condensed unknowns each cell of `sdh` carries, as an iterable of `length(sdh.cellset)`.
Used by `FerriteOperators` to lay out the [`InternalVariableHandler`](@ref). Dispatches on the
element cache, since that is what determines how many condensed unknowns a cell carries.
"""
function FerriteOperators.get_number_of_internal_dofs_per_element(
    integrator,
    element_cache::AnyQuasiStaticElementCache,
    sdh::SubDofHandler,
)
    nqp = getnquadpoints(element_cache.cv)
    # `nothing, nothing`: this asks for *one* size and hands it to every cell and every quadrature point
    # of the subdomain, so it only holds for a material whose local state size is constant. A model whose
    # size varies per point -- FE², see `internal_variable_size` -- cannot be sized here, and the
    # `InternalVariableHandler` block this feeds could not express the result either.
    ndofs_per_qp = internal_variable_size(element_cache.constitutive_model, nothing, nothing)
    return Iterators.repeated(ndofs_per_qp*nqp, length(sdh.cellset))
end

_qs_nbase(e::AnyQuasiStaticElementCache) = getnbasefunctions(e.cv)
# Multiplying one size by the quadrature count assumes every point of the cell carries the same amount of
# state; see `internal_variable_size` for when that stops being true.
_qs_ninternal(e::AnyQuasiStaticElementCache) =
    internal_variable_size(e.constitutive_model, nothing, nothing)*getnquadpoints(e.cv)

"""
    _qs_field_dofs(element_cache, uₑ)

The displacement field's entries of an element-local state vector.

A subdomain declaring [`FerriteOperators.global_dofs`](@ref) -- a chamber pressure tying a surface,
say -- hands its elements the augmented system `[celldofs(cell); global dofs]`. These elements carry
the displacement field alone, so its entries are the head of that vector, which is also the range
`req.r[i]` and `req.K[i, j]` address.
"""
@inline _qs_field_dofs(e::AnyQuasiStaticElementCache, uₑ) = @view uₑ[1:getnbasefunctions(e.cv)]

"""
    _qs_internal_block(element_cache, Qₑ)

View the cell's condensed unknowns as `(size_per_quadrature_point, nquadpoints)`, so a quadrature loop
can address its own slice as `@view Qₑ[:, qp.i]`.

The block is empty for materials without condensed state, in which case the reshape yields a
`0 × nqp` array and every per-point slice is empty.

The reshape is what makes "one size per quadrature point" a structural assumption of the element
layout, not merely of the caller: a material whose state size varies between the points of a cell
(FE², see [`internal_variable_size`](@ref)) has no rectangular block to reshape into. `_qs_ninternal`
sizes the block by multiplying, so the division here is exact by construction.
"""
@inline function _qs_internal_block(e::AnyQuasiStaticElementCache, Qₑ)
    nqp = getnquadpoints(e.cv)
    return reshape(Qₑ, (length(Qₑ) ÷ nqp, nqp))
end

# A weighted sweep that does not name `:u` must contribute nothing from a displacement-only term.
@inline _qs_state_weight(weights::NamedTuple) = haskey(weights, :u) ? weights.u : false

# --- rate-free elements --------------------------------------------------------------------------

FerriteOperators.provides_analytic(::Type{<:QuasiStaticElementCache}, ::FerriteOperators.JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:QuasiStaticElementCache}, ::FerriteOperators.JacobianResidualKind) = true

# TODO how to control dispatch on required input for the material routine?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianResidualRequest,
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = _qs_field_dofs(element_cache, args.states.u)
    time = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # A rate-free scheme has no previous configuration to form a rate from.
        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute stress and tangent
        P, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            time,
        )
        tangent = consistent_tangent(sensitivities)

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            req.r[i] += ∇δui ⊡ P * dΩ

            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                req.K[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function _assemble_quasistatic_jacobian!(req, element_cache::QuasiStaticElementCache, args, w)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = _qs_field_dofs(element_cache, args.states.u)
    time = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute "tangent only"
        _, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            time,
        )
        tangent = consistent_tangent(sensitivities)

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                req.K[i, j] += w * (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

# `true` rather than `1.0`: the unweighted path must not promote the element matrix' eltype.
FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_quasistatic_jacobian!(req, element_cache, args, true)

# A rate-free element is a function of `(u, t)` alone, so only the `:u` weight acts on it.
FerriteOperators.provides_analytic(::Type{<:QuasiStaticElementCache}, ::FerriteOperators.WeightedJacobianKind) = true
FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_quasistatic_jacobian!(req, element_cache, args, _qs_state_weight(req.weights))

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = _qs_field_dofs(element_cache, args.states.u)
    time = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute stress only
        P = reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            time,
        )

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            req.r[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# --- condensed elements --------------------------------------------------------------------------
#
# The scheme hands the element **two** unrelated time quantities, and the context keeps them apart:
#
#  * `stage_scaling(ctx)` -- the interval the *internal variable* integrates over. `dₜQ = L(F, Q)` is
#    first order no matter what the global scheme does with `u`, so its local problem is unchanged by
#    the choice of global scheme.
#  * the `v` slot -- how the deformation rate is formed. `AffineRate` reconstructs it from the
#    unknown displacement at gather time, so the element never learns the scheme's coefficients.
#
# Collapsing the two is what makes a rate-coupled material silently wrong under any scheme but
# backward Euler, where they happen to be reciprocals.

# Forming the rate is the scheme's contribution, and this is the only place in the element layer that
# reads the reconstruction. Dispatching on the cache means the ODE cache does not pay for the second
# gradient evaluation: its local problem `dₜQ = L(F, Q)` cannot read a rate.
@inline function compute_kinematic_quantities(
    e::QuasiStaticCondensedODEElementCache,
    qp,
    dₑ,
    states,
)
    ∇u = function_gradient(e.cv, qp, dₑ)
    return DeformationGradient(one(∇u) + ∇u)
end

@inline function compute_kinematic_quantities(
    e::QuasiStaticCondensedDAEElementCache,
    qp,
    dₑ,
    states,
)
    ∇u = function_gradient(e.cv, qp, dₑ)
    # The `v` slot already carries `∂v∂u (u - uᵥ)`, and `function_gradient` is linear, so its gradient
    # is the deformation rate the material asks for.
    ∇v = function_gradient(e.cv, qp, _qs_field_dofs(e, states.v))
    return DeformationGradientWithRate(one(∇u) + ∇u, ∇v)
end

# How the tangent weighs the two kinematic sensitivities. A reconstructed slot is frozen under
# `JacobianKind{:u}`, so the rate weight is zero there and is the scheme's chain-rule scalar in a
# weighted sweep.
@inline _qs_linearization(::QuasiStaticCondensedODEElementCache, wu, wv) =
    KinematicLinearization(wu)
@inline _qs_linearization(::QuasiStaticCondensedDAEElementCache, wu, wv) =
    KinematicLinearization(wu, wv)

"""
    _qs_rate_weight(weights)

The scheme's `∂v/∂u`, read out of a weighted request's chain-rule scalars.

A weighted sweep over a rate-coupled material must name the `v` slot: without it the local solve's
rate sensitivity would be dropped silently, leaving a descent direction rather than a Newton one.
"""
@inline function _qs_rate_weight(weights::NamedTuple)
    haskey(weights, :v) || throw(ArgumentError(
        "A weighted Jacobian over a rate-coupled material must weigh the `v` slot; got slots " *
        "$(keys(weights)). The rate sensitivity ∂P/∂Ḟ enters through it and through nothing else."))
    return weights.v
end

"""
    _qs_local_state(element_cache, args)

The cell's condensed unknowns and the known state its local problem integrates from, as
`(Qₑ, Qₑprev)` blocks addressable per quadrature point.

`q` is the trial state [`FerriteOperators.condense_internal!`](@ref) wrote; `qprev` is the committed
one. A kernel reads both and writes neither -- the local solve inside `material_routine` re-derives
the same `Q` it was handed, warm started from it, which is what keeps the residual and the tangent
posing one local problem.
"""
@inline function _qs_local_state(e::QuasiStaticCondensedElementCache, args)
    Qₑ     = _qs_internal_block(e, similar(args.states.u, length(args.states.q)))
    Qₑ    .= _qs_internal_block(e, args.states.q)
    Qₑprev = _qs_internal_block(e, args.states.qprev)
    return Qₑ, Qₑprev
end

FerriteOperators.provides_analytic(::Type{<:QuasiStaticCondensedODEElementCache}, ::FerriteOperators.JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:QuasiStaticCondensedODEElementCache}, ::FerriteOperators.JacobianResidualKind) = true
FerriteOperators.provides_analytic(::Type{<:QuasiStaticCondensedDAEElementCache}, ::FerriteOperators.JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:QuasiStaticCondensedDAEElementCache}, ::FerriteOperators.JacobianResidualKind) = true
FerriteOperators.provides_analytic(::Type{<:QuasiStaticCondensedDAEElementCache}, ::FerriteOperators.WeightedJacobianKind) = true

"""
    condense_cell!(cache, args, weights)

Solve every quadrature point's local problem and write the trial internal state into `args.states.q`.

This is the only hook that evolves the condensed state; the assembly kernels are pure evaluations at
the state it wrote.
"""
function FerriteOperators.condense_cell!(
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
    weights::NamedTuple,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ     = _qs_field_dofs(element_cache, args.states.u)
    Qₑ     = _qs_internal_block(element_cache, args.states.q)
    Qₑprev = _qs_internal_block(element_cache, args.states.qprev)
    t      = FerriteOperators.evaluation_time(args.ctx)
    Δt     = FerriteOperators.stage_scaling(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)
        # Through the residual entry point, so that the state written here is the solution of
        # exactly the local problem the assembly kernels evaluate. The stress it returns is the
        # kernels' business, not this hook's.
        reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )
    end
    return zero(FerriteOperators.CondensationReport{Float64})
end

function _assemble_condensed_cell!(
    req,
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
    linearization,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs  = getnbasefunctions(cv)
    dₑ     = _qs_field_dofs(element_cache, args.states.u)
    Qₑ, Qₑprev = _qs_local_state(element_cache, args)
    t      = FerriteOperators.evaluation_time(args.ctx)
    Δt     = FerriteOperators.stage_scaling(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)

        P, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )
        tangent = consistent_tangent(sensitivities, linearization)

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            _qs_accumulate_residual!(req, i, ∇δui ⊡ P * dΩ)

            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                req.K[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

@inline _qs_accumulate_residual!(req::FerriteOperators.JacobianResidualRequest, i, value) =
    (req.r[i] += value)
@inline _qs_accumulate_residual!(req, i, value) = nothing

FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianResidualRequest,
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(req, element_cache, args, _qs_linearization(element_cache, true, false))

FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(req, element_cache, args, _qs_linearization(element_cache, true, false))

FerriteOperators.provides_analytic(::Type{<:QuasiStaticCondensedODEElementCache}, ::FerriteOperators.WeightedJacobianKind) = true
FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticCondensedODEElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(
    req,
    element_cache,
    args,
    _qs_linearization(element_cache, _qs_state_weight(req.weights), false),
)

FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticCondensedDAEElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(
    req,
    element_cache,
    args,
    _qs_linearization(
        element_cache,
        _qs_state_weight(req.weights),
        _qs_rate_weight(req.weights),
    ),
)

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs  = getnbasefunctions(cv)
    dₑ     = _qs_field_dofs(element_cache, args.states.u)
    Qₑ, Qₑprev = _qs_local_state(element_cache, args)
    t      = FerriteOperators.evaluation_time(args.ctx)
    Δt     = FerriteOperators.stage_scaling(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)

        # Stress only
        P = reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            req.r[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# ------------------------------------------------------------------------------------------------

"""
    quasistatic_element_cache_type(evolution::InternalVariableEvolution)

Which quasi-static element cache a material needs, decided by the evolution law of its internal
variable — not by the time integrator. A material with no internal variable, or with one that carries
no time derivative, is rate free; `dₜQ = L(F, Q)` is a mass matrix ODE; `dₜQ = L(F, dₜF, Q)` is a DAE.

The question is asked of the model via [`internal_variable_evolution`](@ref) rather than of the state
cache: the `Empty…CondensationMaterialStateCache` types record only that a model needs no extra
scratch space, which is a different question and gives the wrong answer for an unwrapped rate
dependent model.
"""
quasistatic_element_cache_type(::NoEvolution) = QuasiStaticElementCache
# A steady state material condenses, but its local problem carries no time derivative, so it
# assembles through the rate-free element cache. Whether a cell carries condensed unknowns is decided
# by `internal_variable_size` of the model, not by the element cache type, so this cache serves both
# rows of the rate-free half of the table.
quasistatic_element_cache_type(::SteadyStateEvolution) = QuasiStaticElementCache
quasistatic_element_cache_type(::FirstOrderEvolution) = QuasiStaticCondensedODEElementCache
quasistatic_element_cache_type(::RateCoupledEvolution) = QuasiStaticCondensedDAEElementCache

function setup_quasistatic_element_cache(
    material_model::AbstractMaterialModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
    cv::CellValues,
)
    internal_cache = setup_internal_cache(material_model, qr, sdh)
    return quasistatic_element_cache_type(internal_variable_evolution(material_model))(
        material_model,
        setup_coefficient_cache(material_model, qr, sdh),
        internal_cache,
        cv,
    )
end
function setup_element_cache(model::QuasiStaticModel, qr::QuadratureRule, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    cv         = CellValues(qr, ip, ip_geo)
    return setup_quasistatic_element_cache(model.material_model, qr, sdh, cv)
end

duplicate_for_device(device, model::AbstractMaterialModel) = model
# Reconstruct the *same* concrete cache type, so the problem class survives the move to a device.
function duplicate_for_device(device, cache::AnyQuasiStaticElementCache)
    return (typeof(cache).name.wrapper)(
        duplicate_for_device(device, cache.constitutive_model),
        duplicate_for_device(device, cache.coefficient_cache),
        duplicate_for_device(device, cache.internal_cache),
        duplicate_for_device(device, cache.cv),
    )
end
