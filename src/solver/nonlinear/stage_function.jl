"""
    AbstractSolutionVectorMapping

Bidirectional wiring between two solution vectors laid out by different dof handlers and different
internal variable handlers.

A domain decomposition method (overlapping Schwarz) or a nonlinear preconditioner needs the same
thing to move between a global problem and a local one.

## Interface

    gather!(target, source, mapping)   # source numbering -> target numbering
    scatter!(source, target, mapping)  # target numbering -> source numbering
"""
abstract type AbstractSolutionVectorMapping end

"""
    IdentitySolutionVectorMapping()

The mapping between a numbering and itself. Both directions are no-ops, and a target that aliases the
source is left untouched rather than copied onto itself.
"""
struct IdentitySolutionVectorMapping <: AbstractSolutionVectorMapping end

function gather!(target::AbstractVector, source::AbstractVector, ::IdentitySolutionVectorMapping)
    target === source && return target
    copyto!(target, source)
    return target
end

function scatter!(source::AbstractVector, target::AbstractVector, ::IdentitySolutionVectorMapping)
    source === target && return source
    copyto!(source, target)
    return source
end

"""
    SolutionVectorMapping(dofs, internal_variables)

`dofs[i]` is the source index feeding the target's `i`-th finite element dof, and
`internal_variables[i]` the source index feeding its `i`-th condensed internal variable.

The target's own layout is `[dofs | internal_variables]` because that is the layout every vector a
dof handler assembles into already has -- the condensed unknowns are appended after `ndofs` -- so
this is the target handler's invariant rather than an assumption about the source.
"""
struct SolutionVectorMapping{DofMapType, IVMapType} <: AbstractSolutionVectorMapping
    dofs::DofMapType
    internal_variables::IVMapType
end

function gather!(target::AbstractVector, source::AbstractVector, m::SolutionVectorMapping)
    ndofs = length(m.dofs)
    @inbounds for i ∈ eachindex(m.dofs)
        target[i] = source[m.dofs[i]]
    end
    @inbounds for i ∈ eachindex(m.internal_variables)
        target[ndofs+i] = source[m.internal_variables[i]]
    end
    return target
end

function scatter!(source::AbstractVector, target::AbstractVector, m::SolutionVectorMapping)
    ndofs = length(m.dofs)
    @inbounds for i ∈ eachindex(m.dofs)
        source[m.dofs[i]] = target[i]
    end
    @inbounds for i ∈ eachindex(m.internal_variables)
        source[m.internal_variables[i]] = target[ndofs+i]
    end
    return source
end

"""
    field_dof_mapping(target_dh, target_sym, source_dh, source_sym)

The wiring from every dof of field `target_sym` to the dof of `source_sym` at the same place.

Built by walking the cells and matching `celldofs` entry by entry, rather than computed from an
assumed numbering: the two handlers agree per cell because they carry the same interpolation on the
same cellset, and that is the *only* premise. A dof reached from two cells must resolve to the same
source dof, which is checked rather than assumed -- that check is what turns "the interpolations
match" from a premise into a verified property of the result.
"""
function field_dof_mapping(
    target_dh::Ferrite.AbstractDofHandler,
    target_sym::Symbol,
    source_dh::Ferrite.AbstractDofHandler,
    source_sym::Symbol,
)
    dofs = zeros(Int, ndofs(target_dh))
    for sdh_target ∈ target_dh.subdofhandlers
        # Matched by cellset, not by position: the two handlers are built by separate loops over the
        # same subdomains, and nothing guarantees those loops agree on an order.
        idx = findfirst(sdh -> sdh.cellset == sdh_target.cellset, source_dh.subdofhandlers)
        idx === nothing && error(
            "No subdomain of the source dof handler matches a target subdomain of $(length(sdh_target.cellset)) cells.",
        )
        sdh_source = source_dh.subdofhandlers[idx]
        target_range = dof_range(sdh_target, target_sym)
        source_range = dof_range(sdh_source, source_sym)
        @assert length(target_range) == length(source_range) "Fields $(target_sym) and $(source_sym) do not share an interpolation."
        for cellid ∈ sdh_target.cellset
            target_cdofs = celldofsview(target_dh, cellid)
            source_cdofs = celldofsview(source_dh, cellid)
            for (i, j) ∈ zip(target_range, source_range)
                d, s = target_cdofs[i], source_cdofs[j]
                if dofs[d] == 0
                    dofs[d] = s
                else
                    @assert dofs[d] == s "Inconsistent dof wiring at dof $d: $(dofs[d]) and $s."
                end
            end
        end
    end
    @assert all(>(0), dofs) "Not every dof of $(target_sym) was reached."
    @assert allunique(dofs) "The dof wiring is not injective."
    return dofs
end

"""
    internal_variable_mapping(target_dh, target_lvh, source_dh, source_lvh)

The wiring from every condensed internal variable of the target to its counterpart in the source.

Both handlers lay the condensed unknowns out per cell in the same order -- they are built from the
same integrator over the same grid -- so this matches them cell by cell and errors if a cell carries
a different number of them on the two sides.
"""
function internal_variable_mapping(
    target_dh::Ferrite.AbstractDofHandler,
    target_lvh::InternalVariableHandler,
    source_dh::Ferrite.AbstractDofHandler,
    source_lvh::InternalVariableHandler,
)
    ndofs(target_lvh) == 0 && return Int[]
    # Both sides index their own solution vector, so each handler's block must start where its dof
    # handler ends. Checking it here is what lets the loop below subtract a constant.
    @assert target_lvh.base_offset == ndofs(target_dh)
    @assert source_lvh.base_offset == ndofs(source_dh)
    ncells = getncells(get_grid(target_dh))
    ivs = zeros(Int, ndofs(target_lvh))
    # The ranges are absolute, i.e. indices into the respective solution vector, while `ivs` is
    # indexed within the target's internal variable block.
    for cid = 1:ncells
        target_range = FerriteOperators.internal_variable_range(target_lvh, cid)
        source_range = FerriteOperators.internal_variable_range(source_lvh, cid)
        @assert length(target_range) == length(source_range) "Cell $cid carries a different number of internal variables on the two sides."
        ivs[target_range .- ndofs(target_dh)] .= source_range
    end
    return ivs
end

"""
    StageEvaluation(; slots, p, ctx, weights, condensed)

Everything the operator needs to evaluate one stage besides the stage unknowns themselves.

A scheme states its discretization here once per step, and the nonlinear solver turns it into the
`(states, p, ctx)` triple every FerriteOperators entry point takes:

| field | meaning |
| :---- | :------ |
| `slots` | slot sources other than `:u`, in gather order — `(uprev = …, v = AffineRate(∂v∂u, uᵥ))` |
| `p` | the user parameter bag, i.e. the quantities being optimized. Never time, never history |
| `ctx` | [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)`; `t` is the time the step is solved AT |
| `weights` | the chain-rule scalars of the scheme matrix, `Σₛ wₛ ∂F/∂s` |
| `condensed` | whether the stage vector carries a condensed internal tail |

`weights` is what makes a rate-coupled element assemble its full tangent: a reconstructed slot is
frozen under `∂F/∂u`, so the `∂F/∂v · ∂v/∂u` term reaches the element as `req.weights.v` and nowhere
else. A scheme with no rate names `:u` alone, and the solver then takes the fused Jacobian-residual
route instead of the two weighted sweeps.
"""
struct StageEvaluation{S <: NamedTuple, P, C, W <: NamedTuple}
    slots::S
    p::P
    ctx::C
    weights::W
    condensed::Bool
end
StageEvaluation(;
    slots = (;),
    p = nothing,
    ctx = nothing,
    weights = (u = true,),
    condensed = false,
) = StageEvaluation(slots, p, ctx, weights, condensed)

"""
    THUNDERBOLT_STAGE_SLOTS

The slots every nonlinear operator of this package declares at `setup_operator`.

One declaration serves every scheme: a step passes only the slots its elements read, and declaring a
superset costs nothing because the sweep gathers what `states` carries and no more. Stating it once
is what keeps a scheme from having to know which materials a model happens to contain.
"""
const THUNDERBOLT_STAGE_SLOTS = (:u, :uprev, :q, :qprev, :v)

"""
    AbstractStageFunction

The nonlinear problem one *stage* of a time integration scheme poses.

A stage has its own unknowns, which are neither required to be a subset of the semidiscrete
function's unknowns nor to correspond to any part of the solution vector. It knows how to build them
from the current state and how to write the state back once they are solved for. That is the whole
content of the abstraction, and it is what makes Newmark (which condenses the velocity), an IMEX
split (which condenses the explicitly advanced block) and backward Euler (which condenses nothing)
the same object. A stage may also be *larger* than the state, as a multi-stage scheme's is, which is
why its size is a query rather than `solution_size` of the function.

## Interface

    getoperator(sf)          # the nonlinear operator, *including* the terms the scheme adds
    getfunction(sf)          # the semidiscrete function the stage was built from
    stage_mapping(sf)        # bidirectional wiring to the function's numbering
    stage_parameters(sf)     # the element facing `p` of the current step
    stage_size(sf)           # length of the stage unknown vector
    uncondensed_range(sf)    # the part of it the linear system solves for
    init_stage!(z, sf, u)    # predictor: current state -> stage unknowns
    update_state!(u, sf, z)  # converged stage unknowns -> state, reconstructing what was condensed

The nonlinear solver sees only this. It never learns what time it is -- everything the operator needs
travels in `stage_parameters` -- which is what lets one solver serve a continuation, a time step and,
later, a coupled multi-stage solve.

The stage vector is split by [`uncondensed_range`](@ref): the leading part is what the linear solve
returns an increment for, the rest is condensed at quadrature point level and written by the assembly.
A scheme whose unknowns are laid out differently -- FIRK stacks `s` blocks of `[dofs | internal
variables]` -- says so by overriding that query rather than by matching an unwritten convention.
"""
abstract type AbstractStageFunction end

"""
    uncondensed_range(sf)

The entries of the stage vector the linear system solves for.

Everything outside it is condensed: eliminated at quadrature point level, present in the stage vector
because it is state that has to survive the step, but absent from the global system. `nlsolve!`
subtracts the linear solve's increment over exactly this range.

The default -- a leading block as long as the system -- is the solid mechanics convention, where the
condensed internal variables are appended after the finite element dofs. It is a query rather than an
assumption so that a scheme with a different layout has somewhere to say so.
"""
uncondensed_range(sf::AbstractStageFunction) = Base.OneTo(size(getJ(getoperator(sf)), 1))


stage_mapping(::AbstractStageFunction) = IdentitySolutionVectorMapping()
stage_size(sf::AbstractStageFunction) = solution_size(getfunction(sf))
init_stage!(z::AbstractVector, sf::AbstractStageFunction, u::AbstractVector) =
    gather!(z, u, stage_mapping(sf))
update_state!(u::AbstractVector, sf::AbstractStageFunction, z::AbstractVector) =
    scatter!(u, z, stage_mapping(sf))

# The constraints, the residual norm and the monitors are properties of the *function*; a stage whose
# unknowns carry no constraints of their own forwards rather than reimplementing. Override on a stage
# that constrains its own unknowns differently.
residual_norm(cache::AbstractNonlinearSolverCache, sf::AbstractStageFunction) =
    residual_norm(cache, getfunction(sf))
eliminate_constraints_from_linearization!(
    cache::AbstractNonlinearSolverCache,
    sf::AbstractStageFunction,
) = eliminate_constraints_from_linearization!(cache, getoperator(sf), getfunction(sf))
eliminate_constraints_from_residual!(
    cache::AbstractNonlinearSolverCache,
    sf::AbstractStageFunction,
) = eliminate_constraints_from_residual!(cache, getfunction(sf))
eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    sf::AbstractStageFunction,
    cache::AbstractNonlinearSolverCache,
) = eliminate_constraints_from_increment!(Δu, getfunction(sf), cache)

"""
    FullStateStage(f, op, p)

The stage whose unknowns *are* the function's unknowns: nothing is condensed and nothing is
reconstructed.

Backward Euler and [`HomotopyPathSolver`](@ref) both pose this, and it is what a scheme uses whenever
the quantity it solves for is the whole state. The mapping is the identity and the stage vector
aliases the state, so neither transfer hook copies anything.

`p` is rewritten once per step by the scheme. It is a plain field rather than an argument threaded
through the solver because the operator, not the solver, is what consumes it.
"""
mutable struct FullStateStage{FType, OpType, PType} <: AbstractStageFunction
    const f::FType
    const op::OpType
    p::PType
end

getoperator(sf::FullStateStage) = sf.op
getfunction(sf::FullStateStage) = sf.f
stage_parameters(sf::FullStateStage) = sf.p

"""
    set_stage_parameters!(sf, p)

Hand the stage the element facing parameters of the step about to be solved.

This is the counterpart of upstream `OrdinaryDiffEq` writing `γ`, `c` and `tmp` onto its `NLSolver`
before calling it: the map from stage unknowns back to state, and everything the operator needs to
evaluate, belongs to the stage rather than to the step function.
"""
function set_stage_parameters!(sf::FullStateStage, p)
    sf.p = p
    return sf
end

"""
    update_stage_linearization!(sf, residual, z) -> Bool

Assemble the stage's residual and the matrix its Newton solves with, at the stage unknowns `z`.

Returns `false` when the condensation phase failed to converge, which is a step failure rather than a
Newton failure and is reported by the caller as such.

Two shapes, selected by whether the scheme's [`stage_weights`](@ref) name a slot besides `:u`:

* one fused sweep, `∂F/∂u` and the residual together, when they do not;
* one residual sweep plus one weighted-Jacobian sweep `Σₛ wₛ ∂F/∂s` when they do. A reconstructed
  slot is frozen under `∂F/∂u`, so this is the only route that carries a rate term into the matrix.
"""
function update_stage_linearization!(
    sf::AbstractStageFunction,
    residual::AbstractVector,
    z::AbstractVector,
)
    op      = getoperator(sf)
    states  = stage_states(sf, z)
    p       = stage_user_parameters(sf)
    ctx     = stage_context(sf)
    weights = stage_weights(sf)

    condense_stage!(sf, states) || return false
    if length(weights) == 1
        update_linearization!(op, residual, states, p, ctx)
    else
        evaluate!(op, residual, states, p, ctx)
        assemble_weighted_jacobian!(getJ(op), op, weights, states, p, ctx)
    end
    return true
end

"""
    evaluate_stage_residual!(sf, residual, z) -> Bool

The residual half of [`update_stage_linearization!`](@ref), for a simplified Newton reusing its
Jacobian. The condensation phase still runs: the residual is a function of the condensed state, and
that state is what the phase produces. What a simplified step saves is the tangent sweep — and, with
it, the correctors that only a tangent sweep reads, which is why the phase is elected residual-only
here. The reused Jacobian is the one iteration 0 assembled; it is not re-formed from correctors, so
there is nothing on this route for them to feed.
"""
function evaluate_stage_residual!(
    sf::AbstractStageFunction,
    residual::AbstractVector,
    z::AbstractVector,
)
    states = stage_states(sf, z)
    condense_stage!(sf, states, nothing) || return false
    evaluate!(getoperator(sf), residual, states, stage_user_parameters(sf), stage_context(sf))
    return true
end

"""
    rollback_stage!(u, uprev, sf)

Restore the committed solution and invalidate the correctors the condensation phase stored for the
trial being discarded. A stage that condenses nothing has none, so it only copies.
"""
function rollback_stage!(u::AbstractVector, uprev::AbstractVector, sf::AbstractStageFunction)
    stage_is_condensed(sf) || return (u .= uprev; u)
    return FerriteOperators.rollback_state!(condensed_operator(getoperator(sf)), u, uprev)
end

"""
    condensed_operator(op)

The `FerriteOperators` operator whose element caches carry the condensation correctors. A scheme
operator that wraps another forwards to it, exactly as it forwards [`getJ`](@ref).
"""
condensed_operator(op) = op

"""
    condense_stage!(sf, states, weights = stage_weights(sf)) -> Bool

Solve every element's local problem and write the trial internal state — its own domain traversal,
after which the assembly sweeps are pure evaluations at the state it wrote.

`weights = nothing` elects a residual-only condensation: `q` is solved for exactly as before, but no
tangent corrector is formed and the stored ones are dropped. A residual sweep reads neither, so this
is the shape [`evaluate_stage_residual!`](@ref) wants; a tangent sweep at that state then throws
rather than combining a previous trial point's corrections.

Returns `false` when a local problem did not converge, having described the failure — this is the one
step failure that no residual norm can be quoted for, since the sweeps that would have filled the
residual never ran.
"""
function condense_stage!(sf::AbstractStageFunction, states::NamedTuple, weights = stage_weights(sf))
    stage_is_condensed(sf) || return true
    report = condense_internal!(
        getoperator(sf),
        weights,
        states,
        stage_user_parameters(sf),
        stage_context(sf),
    )
    report.converged && return true
    # The report's argmax carriers are all that survives the fold, and they are what names the
    # offender. A solver holding the per-point store can say more; see the multilevel Newton.
    @debug "Local solve did not converge. Aborting. $report" _group = :nlsolve
    return false
end

"""
    stage_states(sf, z)

The slot sources of the current step, with the stage unknowns bound to `:u`.

`:u` leads, which is the framework's requirement for any [`AffineRate`](@ref) slot reconstructed from
it. The condensed internal state is sourced from the same vector: `z` is laid out
`[dofs | internal variables]`, and an `InternalSource` restricts the gather to that tail.
"""
function stage_states(sf::AbstractStageFunction, z::AbstractVector)
    e = stage_parameters(sf)
    states = merge((u = z,), e.slots)
    return e.condensed ? merge(states, (q = InternalSource(z),)) : states
end

stage_context(sf::AbstractStageFunction) = stage_parameters(sf).ctx
stage_user_parameters(sf::AbstractStageFunction) = stage_parameters(sf).p
stage_weights(sf::AbstractStageFunction) = stage_parameters(sf).weights
stage_is_condensed(sf::AbstractStageFunction) = stage_parameters(sf).condensed
