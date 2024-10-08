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
    subintsetType
} <: DiffEqBase.AbstractODEIntegrator{algType, true, uType, tType}
    f::fType
    alg::algType
    u::uType # Master Solution
    uprev::uType # Master Solution
    p::pType
    t::tType # Current time
    tprev::tType
    dt::tType # This is the time step length which which we intend to advance
    _dt::tType # This is the time step length which which we use during time marching
    dtchangeable::Bool # Indicator whether _dt can be changed
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
    subintegrators::subintsetType
end

# called by DiffEqBase.init and DiffEqBase.solve
function DiffEqBase.__init(
    prob::OperatorSplittingProblem,
    alg::AbstractOperatorSplittingAlgorithm,
    args...;
    dt,
    tstops = (),
    saveat = nothing,
    save_everystep = false,
    callback = nothing,
    advance_to_tstop = false,
    save_func = (u, t) -> copy(u), # custom kwarg
    dtchangeable = DiffEqBase.isadaptive(alg),           # custom kwarg
    kwargs...,
)
    (; u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    dt = tf > t0 ? dt : -dt

    _tstops = tstops
    _saveat = saveat
    tstops, saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)

    sol = DiffEqBase.build_solution(prob, alg, typeof(t0)[], typeof(save_func(u0, t0))[])

    callback = DiffEqBase.CallbackSet(callback)

    cache = init_cache(prob, alg; u0, t0, dt, kwargs...)

    subintegrators = build_subintegrators_recursive(prob.f, prob.f.synchronizers, p, cache, t0, dt, 1:length(u0), cache.u, tstops, _tstops, saveat, _saveat)

    integrator = OperatorSplittingIntegrator(
        prob.f,
        alg,
        cache.u,
        cache.uprev,
        p,
        t0,
        copy(t0),
        dt,
        _dt,
        dtchangeable,
        tstops,
        _tstops,
        saveat,
        _saveat,
        callback,
        advance_to_tstop,
        false,
        cache,
        sol,
        subintegrators,
    )
    DiffEqBase.initialize!(callback, u0, t0, integrator) # Do I need this?
    return integrator
end

DiffEqBase.has_reinit(integrator::OperatorSplittingIntegrator) = true
function DiffEqBase.reinit!(
    integrator::OperatorSplittingIntegrator,
    u0 = integrator.sol.prob.u0;
    tspan = integrator.sol.prob.tspan,
    erase_sol = true,
    tstops = integrator._tstops,
    saveat = integrator._saveat,
    reinit_callbacks = true,
    reinit_retcode = true
)
    (t0,tf) = tspan
    integrator.u .= u0
    integrator.t = t0
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
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, DiffEqBase.ReturnCode.Default)
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
        DiffEqBase.check_error!(integrator) ∉ (DiffEqBase.ReturnCode.Success, DiffEqBase.ReturnCode.Default) && return
        __step!(integrator)
    end
    DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    if DiffEqBase.NAN_CHECK(integrator.u)
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, DiffEqBase.ReturnCode.Failure)
    else
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, DiffEqBase.ReturnCode.Success)
    end
    return integrator.sol
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator)
    if integrator.advance_to_tstop
        tstop = first(integrator.tstops)
        while !reached_tstop(integrator, tstop)
            DiffEqBase.check_error!(integrator) ∉ (DiffEqBase.ReturnCode.Success, DiffEqBase.ReturnCode.Default) && return
            __step!(integrator)
        end
    else
        DiffEqBase.check_error!(integrator) ∉ (DiffEqBase.ReturnCode.Success, DiffEqBase.ReturnCode.Default) && return
        __step!(integrator)
    end
end

function DiffEqBase.check_error!(integrator::OperatorSplittingIntegrator)
    if DiffEqBase.NAN_CHECK(integrator._dt) # replace with https://github.com/SciML/OrdinaryDiffEq.jl/blob/373a8eec8024ef1acc6c5f0c87f479aa0cf128c3/lib/OrdinaryDiffEqCore/src/iterator_interface.jl#L5-L6 after moving to sciml integrators
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, DiffEqBase.ReturnCode.Failure)
    end
    return integrator.sol.retcode
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator, dt, stop_at_tdt = false)
    # OridinaryDiffEq lets dt be negative if tdir is -1, but that's inconsistent
    dt <= zero(dt) && error("dt must be positive")
    stop_at_tdt && !integrator.dtchangeable && error("Cannot stop at t + dt if dtchangeable is false")
    tnext = integrator.t + tdir(integrator) * dt
    stop_at_tdt && DiffEqBase.add_tstop!(integrator, tnext)
    while !reached_tstop(integrator, tnext, stop_at_tdt)
        DiffEqBase.check_error!(integrator) ∉ (DiffEqBase.ReturnCode.Success, DiffEqBase.ReturnCode.Default) && return
        __step!(integrator)
    end
end

# TimeChoiceIterator API
@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator)
    DiffEqBase.get_tmp_cache(integrator, integrator.alg, integrator.cache)
end
@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator, ::AbstractOperatorSplittingAlgorithm, cache)
    return (cache.tmp,)
end
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
        return integrator.t == tstop # Check for exact hit
    else #!stop_at_tstop
        return is_past_t(integrator, tstop)
    end
end



# Dunno stuff
function DiffEqBase.SciMLBase.done(integrator::OperatorSplittingIntegrator)
    if !(integrator.sol.retcode in (DiffEqBase.ReturnCode.Default, DiffEqBase.ReturnCode.Success))
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

    # Propagate information down into the subintegrators
    synchronize_subintegrators!(integrator)
    tnext = integrator.t + integrator.dt

    # Solve inner problems
    advance_solution_to!(integrator, tnext; uparent=integrator.u)
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
function advance_solution_to!(integrator, tnext; uparent)
    advance_solution_to!(integrator, integrator.cache, tnext; uparent)
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

function synchronize_subintegrators!(integrator::OperatorSplittingIntegrator)
    synchronize_subintegrator!(integrator.subintegrators, integrator)
end

@unroll function synchronize_subintegrator!(subintegrators::Tuple, integrator::OperatorSplittingIntegrator)
    @unroll for subintegrator in subintegrators
        synchronize_subintegrator!(subintegrator, integrator)
    end
end

function advance_solution_to!(integrator::OperatorSplittingIntegrator, cache::AbstractOperatorSplittingCache, tnext::Number; uparent)
    advance_solution_to!(integrator.subintegrators, cache, tnext; uparent)
end

# Dispatch for tree node construction
function build_subintegrators_recursive(f::GenericSplitFunction, synchronizers::Tuple, p::Tuple, cache::AbstractOperatorSplittingCache, t, dt, dof_range, uparent, tstops, _tstops, saveat, _saveat)
    return ntuple(i ->
        build_subintegrators_recursive(
            get_operator(f, i),
            synchronizers[i],
            p[i],
            cache.inner_caches[i],
            # TODO recover this
            t, dt, f.dof_ranges[i],
            # We pass the full solution, because some parameters might require
            # access to solution variables which are not part of the local solution range
            uparent,
            tstops, _tstops, saveat, _saveat
        ), length(f.functions)
    )
end
function build_subintegrators_recursive(f::GenericSplitFunction, synchronizers::NoExternalSynchronization, p::Tuple, cache::AbstractOperatorSplittingCache, t, dt, dof_range, uparent, tstops, _tstops, saveat, _saveat)
    return ntuple(i ->
        build_subintegrators_recursive(
            get_operator(f, i),
            synchronizers,
            p[i],
            cache.inner_caches[i],
            # TODO recover this
            t, dt, f.dof_ranges[i],
            # We pass the full solution, because some parameters might require
            # access to solution variables which are not part of the local solution range
            uparent,
            tstops, _tstops, saveat, _saveat
        ), length(f.functions)
    )
end

@unroll function prepare_local_step!(uparent, subintegrators::Tuple)
    @unroll for subintegrator in subintegrators
        prepare_local_step!(uparent, subintegrator)
    end
end

@unroll function finalize_local_step!(uparent, subintegrators::Tuple)
    @unroll for subintegrator in subintegrators
        finalize_local_step!(uparent, subintegrator)
    end
end
