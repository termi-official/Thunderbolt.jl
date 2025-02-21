"""
    OperatorSplittingIntegrator <: AbstractODEIntegrator

A variant of [`ODEIntegrator`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123) to perform opeartor splitting.

Derived from https://github.com/CliMA/ClimaTimeSteppers.jl/blob/ef3023747606d2750e674d321413f80638136632/src/integrators.jl.
"""
mutable struct OperatorSplittingIntegrator{
    fType,
    algType,
    uType,
    tType,
    pType,
    heapType,
    tstopsType,
    saveatType,
    callbackType,
    cacheType,
    solType,
    subintTreeType,
    solidxTreeType,
    syncTreeType,
} <: DiffEqBase.AbstractODEIntegrator{algType, true, uType, tType}
    const f::fType
    const alg::algType
    u::uType # Master Solution
    uprev::uType # Master Solution
    tmp::uType # Interpolation buffer
    p::pType
    t::tType # Current time
    tprev::tType
    dt::tType # This is the time step length which which we intend to advance
    _dt::tType # This is the time step length which which we use during time marching
    const dtchangeable::Bool # Indicator whether _dt can be changed
    tstops::heapType
    _tstops::tstopsType # argument to __init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType # argument to __init used as default argument to reinit!
    callback::callbackType
    advance_to_tstop::Bool
    u_modified::Bool # not used; field is required for compatibility with
    # DiffEqBase.initialize! and DiffEqBase.finalize!
    cache::cacheType
    sol::solType
    subintegrator_tree::subintTreeType
    solution_index_tree::solidxTreeType
    synchronizer_tree::syncTreeType
end

# called by DiffEqBase.init and DiffEqBase.solve
function DiffEqBase.__init(
    prob::OperatorSplittingProblem,
    alg::AbstractOperatorSplittingAlgorithm,
    args...;
    dt,
    tstops = (),
    saveat = (),
    d_discontinuities = (),
    save_everystep = false,
    callback = nothing,
    advance_to_tstop = false,
    adaptive = DiffEqBase.isadaptive(alg),
    controller = nothing,
    alias_u0 = true,
    verbose = true,
    kwargs...,
)
    (; u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    dt = tf > t0 ? dt : -dt
    tType = typeof(dt)

    dtchangeable = DiffEqBase.isadaptive(alg)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, prob.tspan)
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, prob.tspan)
    d_discontinuities_internal = OrdinaryDiffEqCore.initialize_d_discontinuities(tType, d_discontinuities, prob.tspan)

    u     = setup_u(prob, alg, alias_u0)
    uprev = setup_u(prob, alg, false)
    tmp   = setup_u(prob, alg, false)
    uType                = typeof(u)

    sol = DiffEqBase.build_solution(prob, alg, tType[], uType[])

    callback = DiffEqBase.CallbackSet(callback)

    # There should be a dispatch which calls some __init dispatch on the leaf algorithms
    subintegrator_tree, cache = build_subintegrator_tree_with_cache(
        prob, alg,
        uprev, u,
        1:length(u),
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose,
    )

    integrator = OperatorSplittingIntegrator(
        prob.f,
        alg,
        u,
        uprev,
        tmp,
        p,
        t0,
        copy(t0),
        dt,
        _dt,
        dtchangeable,
        tstops_internal,
        tstops,
        saveat_internal,
        saveat,
        callback,
        advance_to_tstop,
        false,
        cache,
        sol,
        subintegrator_tree,
        build_solution_index_tree(prob.f),
        build_synchronizer_tree(prob.f),
    )
    DiffEqBase.initialize!(callback, u0, t0, integrator) # Do I need this?
    return integrator
end

DiffEqBase.has_reinit(integrator::OperatorSplittingIntegrator) = true
function DiffEqBase.reinit!(
    integrator::OperatorSplittingIntegrator,
    u0 = integrator.sol.prob.u0;
    t0 = integrator.sol.prob.tspan[1],
    tf = integrator.sol.prob.tspan[2],
    erase_sol = false,
    tstops = integrator._tstops,
    saveat = integrator._saveat,
    reinit_callbacks = true,
    reinit_retcode = true
)
    integrator.u .= u0
    integrator.uprev .= u0
    integrator.t = t0
    integrator.tprev = t0
    integrator.tstops, integrator.saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        DiffEqBase.initialize!(integrator.callback, u0, t0, integrator)
    else # always reinit the saving callback so that t0 can be saved if needed
        saving_callback = integrator.callback.discrete_callbacks[end]
        DiffEqBase.initialize!(saving_callback, u0, t0, integrator)
    end
    if reinit_retcode
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Default)
    end

    subreinit!(
        integrator.subintegrator_tree;
        t0, tf,
        erase_sol,
        tstops,
        saveat,
        reinit_callbacks,
        reinit_retcode,
    )
end

function subreinit!(
    subintegrator::DiffEqBase.DEIntegrator;
    kwargs...
)
    DiffEqBase.reinit!(subintegrator, subintegrator.uprev; kwargs...)
end

function subreinit!(
    subintegrators::Tuple;
    kwargs...
)
    for subintegrator in subintegrators
        subreinit!(subintegrator; kwargs...)
    end
end

# called by DiffEqBase.solve
function DiffEqBase.__solve(prob::OperatorSplittingProblem, alg::AbstractOperatorSplittingAlgorithm, args...; kwargs...)
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    DiffEqBase.solve!(integrator)
end

# either called directly (after init), or by DiffEqBase.solve (via __solve)
function DiffEqBase.solve!(integrator::OperatorSplittingIntegrator)
    while !isempty(integrator.tstops)
        @timeit_debug "check_error" DiffEqBase.check_error!(integrator) ∉ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default) && return
        __step!(integrator)
    end
    DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    if DiffEqBase.NAN_CHECK(integrator.u)
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Unstable)
    else
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Success)
    end
    return integrator.sol
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator)
    @timeit_debug "step!" if integrator.advance_to_tstop
        tstop = first(integrator.tstops)
        while !reached_tstop(integrator, tstop)
            @timeit_debug "check_error" DiffEqBase.check_error!(integrator) ∉ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default) && return
            __step!(integrator)
        end
    else
        @timeit_debug "check_error" DiffEqBase.check_error!(integrator) ∉ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default) && return
        __step!(integrator)
    end
end

function SciMLBase.check_error(integrator::OperatorSplittingIntegrator)
    if !SciMLBase.successful_retcode(integrator.sol) && integrator.sol.retcode != SciMLBase.ReturnCode.Default
        return integrator.sol.retcode
    end

    verbose = true # integrator.opts.verbose

    if DiffEqBase.NAN_CHECK(integrator._dt) # replace with https://github.com/SciML/OrdinaryDiffEq.jl/blob/373a8eec8024ef1acc6c5f0c87f479aa0cf128c3/lib/OrdinaryDiffEqCore/src/iterator_interface.jl#L5-L6 after moving to sciml integrators
        if verbose
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        end
        return SciMLBase.ReturnCode.DtNaN
    end

    return check_error_subintegrators(integrator, integrator.subintegrator_tree)
end

function check_error_subintegrators(integrator, subintegrator_tree::Tuple)
    for subintegrator in subintegrator_tree
        retcode = check_error_subintegrators(integrator, subintegrator)
        if !SciMLBase.successful_retcode(retcode) && retcode != SciMLBase.ReturnCode.Default
            return retcode
        end
    end
    return integrator.sol.retcode
end

function check_error_subintegrators(integrator, subintegrator::SciMLBase.DEIntegrator)
    return SciMLBase.check_error(subintegrator)
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator, dt, stop_at_tdt = false)
    @timeit_debug "step!" begin
    # OridinaryDiffEq lets dt be negative if tdir is -1, but that's inconsistent
    dt <= zero(dt) && error("dt must be positive")
    stop_at_tdt && !integrator.dtchangeable && error("Cannot stop at t + dt if dtchangeable is false")
    tnext = integrator.t + tdir(integrator) * dt
    stop_at_tdt && DiffEqBase.add_tstop!(integrator, tnext)
    while !reached_tstop(integrator, tnext, stop_at_tdt)
        @timeit_debug "check_error" DiffEqBase.check_error!(integrator) ∉ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default) && return
        __step!(integrator)
    end
    end
end

function setup_u(prob::OperatorSplittingProblem, solver, alias_u0)
    if alias_u0
        return prob.u0
    else
        return OrdinaryDiffEqCore.recursivecopy(prob.u0)
    end
end

# TimeChoiceIterator API
@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator)
    # DiffEqBase.get_tmp_cache(integrator, integrator.alg, integrator.cache)
    (integrator.tmp,)
end
# @inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator, ::AbstractOperatorSplittingAlgorithm, cache)
#     return (cache.tmp,)
# end
# Interpolation
# TODO via https://github.com/SciML/SciMLBase.jl/blob/master/src/interpolation.jl
function linear_interpolation!(y,t,y1,y2,t1,t2)
    y .= y1 + (t-t1) * (y2-y1)/(t2-t1)
end
function (integrator::OperatorSplittingIntegrator)(tmp, t)
    linear_interpolation!(tmp, t, integrator.uprev, integrator.u, integrator.tprev, integrator.t)
end

"""
    stepsize_controller!(::OperatorSplittingIntegrator)
Updates the controller using the current state of the integrator if the operator splitting algorithm is adaptive.
"""
@inline function stepsize_controller!(integrator::OperatorSplittingIntegrator) 
    algorithm = integrator.alg
    DiffEqBase.isadaptive(algorithm) || return nothing
    stepsize_controller!(integrator, algorithm)
end

"""
    step_accept_controller!(::OperatorSplittingIntegrator)
Updates `_dt` of the integrator if the step is accepted and the operator splitting algorithm is adaptive.
"""
@inline function step_accept_controller!(integrator::OperatorSplittingIntegrator) 
    algorithm = integrator.alg
    DiffEqBase.isadaptive(algorithm) || return nothing
    step_accept_controller!(integrator, algorithm, nothing)
end

"""
    step_reject_controller!(::OperatorSplittingIntegrator)
Updates `_dt` of the integrator if the step is rejected and the the operator splitting algorithm is adaptive.
"""
@inline function step_reject_controller!(integrator::OperatorSplittingIntegrator) 
    algorithm = integrator.alg
    DiffEqBase.isadaptive(algorithm) || return nothing
    step_reject_controller!(integrator, algorithm, nothing)
end

# helper functions for dealing with time-reversed integrators in the same way
# that OrdinaryDiffEq.jl does
tdir(integrator) = integrator.tstops.ordering isa DataStructures.FasterForward ? 1 : -1
is_past_t(integrator, t) = tdir(integrator) * (t - integrator.t) ≤ zero(integrator.t)
function reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable)
    if stop_at_tstop
        integrator.t > tstop && error("Integrator missed stop at $tstop (current time=$(integrator.t)). Aborting.")
        return integrator.t == tstop # Check for exact hit
    else #!stop_at_tstop
        return is_past_t(integrator, tstop)
    end
end



# Dunno stuff
function SciMLBase.done(integrator::OperatorSplittingIntegrator)
    if !(integrator.sol.retcode in (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success))
        return true
    elseif isempty(integrator.tstops)
        DiffEqBase.postamble!(integrator)
        return true
    # elseif integrator.just_hit_tstop
        # integrator.just_hit_tstop = false
        # if integrator.opts.stop_at_next_tstop
        #     postamble!(integrator)
        #     return true
        # end
    # else
        # @error "What to do here?"
    end
    false
end

function DiffEqBase.postamble!(integrator::OperatorSplittingIntegrator)
    DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
end

function __step!(integrator)
    (; dtchangeable, tstops) = integrator
    _dt = DiffEqBase.get_dt(integrator)

    # update dt before incrementing u; if dt is changeable and there is
    # a tstop within dt, reduce dt to tstop - t
    integrator.dt =
        !isempty(tstops) && dtchangeable ? tdir(integrator) * min(_dt, abs(first(tstops) - integrator.t)) :
        tdir(integrator) * _dt

    # Propagate information down into the subintegrator_tree
    synchronize_subintegrator_tree!(integrator)
    tnext = integrator.t + integrator.dt

    # Solve inner problems
    advance_solution_to!(integrator, tnext)
    stepsize_controller!(integrator)

    # Update integrator
    # increment t by dt, rounding to the first tstop if that is roughly
    # equivalent up to machine precision; the specific bound of 100 * eps...
    # is taken from OrdinaryDiffEq.jl
    t_unit = oneunit(integrator.t)
    max_t_error = 100 * eps(float(integrator.t / t_unit)) * t_unit
    integrator.tprev = integrator.t
    integrator.t = !isempty(tstops) && abs(first(tstops) - tnext) < max_t_error ? first(tstops) : tnext

    step_accept_controller!(integrator)

    # remove tstops that were just reached
    while !isempty(tstops) && reached_tstop(integrator, first(tstops))
        pop!(tstops)
    end
end

# solvers need to define this interface
function advance_solution_to!(integrator::OperatorSplittingIntegrator, tnext)
    advance_solution_to!(integrator, integrator.cache, tnext)
end

function advance_solution_to!(outer_integrator::OperatorSplittingIntegrator, integrator::DiffEqBase.DEIntegrator, solution_indices, sync, cache, tend)
    dt = tend-integrator.t
    SciMLBase.step!(integrator, dt, true)
end

DiffEqBase.get_dt(integrator::OperatorSplittingIntegrator) = integrator._dt
function set_dt!(integrator::OperatorSplittingIntegrator, dt)
    # TODO: figure out interface for recomputing other objects (linear operators, etc)
    dt <= zero(dt) && error("dt must be positive")
    integrator._dt = dt
end

function DiffEqBase.add_tstop!(integrator::OperatorSplittingIntegrator, t)
    is_past_t(integrator, t) && error("Cannot add a tstop at $t because that is behind the current \
                                       integrator time $(integrator.t)")
    push!(integrator.tstops, t)
end

function DiffEqBase.add_saveat!(integrator::OperatorSplittingIntegrator, t)
    is_past_t(integrator, t) && error("Cannot add a saveat point at $t because that is behind the \
                                       current integrator time $(integrator.t)")
    push!(integrator.saveat, t)
end

# not sure what this should do?
# defined as default initialize: https://github.com/SciML/DiffEqBase.jl/blob/master/src/callbacks.jl#L3
DiffEqBase.u_modified!(i::OperatorSplittingIntegrator, bool) = nothing

function synchronize_subintegrator_tree!(integrator::OperatorSplittingIntegrator)
    synchronize_subintegrator!(integrator.subintegrator_tree, integrator)
end

@unroll function synchronize_subintegrator!(subintegrator_tree::Tuple, integrator::OperatorSplittingIntegrator)
    @unroll for subintegrator in subintegrator_tree
        synchronize_subintegrator!(subintegrator, integrator)
    end
end

function synchronize_subintegrator!(subintegrator::SciMLBase.DEIntegrator, integrator::OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    @assert subintegrator.t == t
    if !DiffEqBase.isadaptive(subintegrator)
        SciMLBase.set_proposed_dt!(subintegrator, dt)
    end
end

function advance_solution_to!(integrator::OperatorSplittingIntegrator, cache::AbstractOperatorSplittingCache, tnext::Number)
    advance_solution_to!(integrator, integrator.subintegrator_tree, integrator.solution_index_tree, integrator.synchronizer_tree, cache, tnext)
end

# Dispatch for tree node construction
function build_subintegrator_tree_with_cache(
    prob::OperatorSplittingProblem, alg::AbstractOperatorSplittingAlgorithm,
    uprevouter::AbstractVector, uouter::AbstractVector,
    solution_indices,
    t0, dt, tf,
    tstops, saveat, d_discontinuities, callback,
    adaptive, verbose,
)
    (; f, p) = prob
    subintegrator_tree_with_caches = ntuple(i ->
        build_subintegrator_tree_with_cache(
            get_operator(f, i),
            alg.inner_algs[i],
            p[i],
            uprevouter, uouter,
            f.solution_indices[i],
            t0, dt, tf,
            tstops, saveat, d_discontinuities, callback,
            adaptive, verbose,
        ),
        length(f.functions)
    )

    subintegrator_tree = ntuple(i -> subintegrator_tree_with_caches[i][1], length(f.functions))
    caches         = ntuple(i -> subintegrator_tree_with_caches[i][2], length(f.functions))

    # TODO fix mixed device type problems we have to be smarter
    return subintegrator_tree, init_cache(f, alg;
        uprev=uprevouter, u=uouter, alias_u=true,
        inner_caches = caches,
    )
end

function build_subintegrator_tree_with_cache(
    f::GenericSplitFunction, alg::AbstractOperatorSplittingAlgorithm, p::Tuple,
    uprevouter::AbstractVector, uouter::AbstractVector,
    solution_indices,
    t0, dt, tf,
    tstops, saveat, d_discontinuities, callback,
    adaptive, verbose,
)
    subintegrator_tree_with_caches = ntuple(i ->
        build_subintegrator_tree_with_cache(
            get_operator(f, i),
            alg.inner_algs[i],
            p[i],
            uprevouter, uouter,
            f.solution_indices[i],
            t0, dt, tf,
            tstops, saveat, d_discontinuities, callback,
            adaptive, verbose,
        ),
        length(f.functions)
    )

    subintegrator_tree = first.(subintegrator_tree_with_caches)
    inner_caches   = last.(subintegrator_tree_with_caches)

    # TODO fix mixed device type problems we have to be smarter
    uprev = @view uprevouter[solution_indices]
    u     = @view uouter[solution_indices]
    return subintegrator_tree, init_cache(f, alg;
        uprev = uprev, u = u,
        inner_caches = inner_caches,
    )
end

forward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator, subintegrator_tree::Tuple, solution_indices::Tuple, synchronizers::Tuple) = nothing
backward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator, subintegrator_tree::Tuple, solution_indices::Tuple) = nothing
