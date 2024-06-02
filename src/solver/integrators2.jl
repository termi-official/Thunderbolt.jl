module OS

import Unrolled: @unroll

using TimerOutputs
# import Test: @inferred

import DiffEqBase, DataStructures

import UnPack: @unpack
import DiffEqBase: ODEFunction, init, TimeChoiceIterator

abstract type AbstractOperatorSplittingAlgorithm end

abstract type AbstractOperatorSplitFunction <: DiffEqBase.AbstractODEFunction{true} end

"""
    GenericSplitFunction(functions::Tuple, dof_ranges::AbstractVector)

This type of function describes a set of connected inner functions in mass-matrix form, as usually found in operator splitting procedures.
"""
struct GenericSplitFunction{fSetType <: Tuple, idxSetType <: AbstractVector} <: AbstractOperatorSplitFunction
    # The atomic ode functions
    functions::fSetType
    # The ranges for the values in the solution vector.
    dof_ranges::idxSetType
end

@inline get_operator(f::GenericSplitFunction, i::Integer) = f.functions[i]

recursive_null_parameters(f::AbstractOperatorSplitFunction) = @error "Not implemented"
recursive_null_parameters(f::GenericSplitFunction) = ntuple(i->recursive_null_parameters(get_operator(f, i)), length(f.functions));
recursive_null_parameters(f::DiffEqBase.AbstractODEFunction) = DiffEqBase.NullParameters()

"""
    OperatorSplittingProblem(f::AbstractOperatorSplitFunction, u0, tspan, p::Tuple)
"""
mutable struct OperatorSplittingProblem{fType <: AbstractOperatorSplitFunction, uType, tType, pType <: Tuple, K} <: DiffEqBase.AbstractODEProblem{uType, tType, true}
    f::fType
    u0::uType
    tspan::tType
    p::pType
    kwargs::K
    function OperatorSplittingProblem(f::AbstractOperatorSplitFunction,
        u0, tspan, p = recursive_null_parameters(f);
        kwargs...)
        new{typeof(f),typeof(u0),
        typeof(tspan), typeof(p),
        typeof(kwargs)}(f,
        u0,
        tspan,
        p,
        kwargs)
    end
end

"""
Internal helper to integrate a single inner operator
over some time interval.
"""
mutable struct ThunderboltSubIntegrator{
    fType,
    uType,
    uprevType,
    tType,
    pType,
}  #<: DiffEqBase.AbstractODEIntegrator{algType, true, uType, tType}
    f::fType # Right hand side
    u::uType # Current solution
    uprev::uprevType
    p::pType
    t::tType
    dt::tType
end

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
    dt::tType
    _dt::tType # argument to __init used to set dt in step!
    dtchangeable::Bool
    tstops::heapType
    _tstops::tstopsType # argument to __init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType # argument to __init used as default argument to reinit!
    step::Int
    stepstop::Int
    callback::callbackType
    advance_to_tstop::Bool
    u_modified::Bool # not used; field is required for compatibility with
    # DiffEqBase.initialize! and DiffEqBase.finalize!
    cache::cacheType
    sol::solType#
    subintegrators::subintsetType
end

# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? DataStructures.FasterForward : DataStructures.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = DataStructures.BinaryHeap{FT, ordering}(tstops)

    if isnothing(saveat)
        saveat = [t0, tf]
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        saveat = tf > t0 ? saveat : -saveat
        saveat = [t0:saveat:tf..., tf]
    else
        # We do not need to filter saveat like tstops because the saving
        # callback will ignore any times that are not between t0 and tf.
        saveat = collect(saveat)
    end
    saveat = DataStructures.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
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
    dtchangeable = true,           # custom kwarg
    stepstop = -1,                 # custom kwarg
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

    cache = init_cache(prob, alg; dt, kwargs...)
    init_cache(prob, alg; dt, kwargs...)

    subintegrators = build_subintegrators_recursive(prob.f, p, cache, cache.u, cache.uprev, t0, dt)
    build_subintegrators_recursive(prob.f, p, cache, cache.u, cache.uprev, t0, dt)

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
        0,
        stepstop,
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
)
    (t0,tf) = tspan
    integrator.u .= u0
    integrator.t = t0
    integrator.tstops, integrator.saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    integrator.step = 0
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
end

# called by DiffEqBase.solve
function DiffEqBase.__solve(prob::OperatorSplittingProblem, alg::AbstractOperatorSplittingAlgorithm, args...; kwargs...)
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    DiffEqBase.solve!(integrator)
end

# either called directly (after init), or by DiffEqBase.solve (via __solve)
function DiffEqBase.solve!(integrator::OperatorSplittingIntegrator)
    while !isempty(integrator.tstops) && integrator.step != integrator.stepstop
        __step!(integrator)
    end
    DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    return integrator.sol
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator)
    if integrator.advance_to_tstop
        tstop = first(integrator.tstops)
        while !reached_tstop(integrator, tstop)
            __step!(integrator)
        end
    else
        __step!(integrator)
    end
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator, dt, stop_at_tdt = false)
    # OridinaryDiffEq lets dt be negative if tdir is -1, but that's inconsistent
    dt <= zero(dt) && error("dt must be positive")
    stop_at_tdt && !integrator.dtchangeable && error("Cannot stop at t + dt if dtchangeable is false")
    t_plus_dt = integrator.t + tdir(integrator) * dt
    stop_at_tdt && DiffEqBase.add_tstop!(integrator, t_plus_dt)
    while !reached_tstop(integrator, t_plus_dt, stop_at_tdt)
        __step!(integrator)
    end
end



# TimeChoiceIterator API
@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator)
    DiffEqBase.get_tmp_cache(integrator, integrator.alg, integrator.cache)
end
# Interpolation
# TODO via https://github.com/SciML/SciMLBase.jl/blob/master/src/interpolation.jl
function linear_interpolation!(y,t,y1,y2,t1,t2)
    y .= y1 + (t-t1) * (y2-y1)/(t2-t1)
end
function (integrator::OperatorSplittingIntegrator)(tmp, t)
    linear_interpolation!(tmp, t, integrator.cache.uprev, integrator.u, integrator.t-integrator.dt, integrator.t)
end



# helper functions for dealing with time-reversed integrators in the same way
# that OrdinaryDiffEq.jl does
tdir(integrator) = integrator.tstops.ordering isa DataStructures.FasterForward ? 1 : -1
is_past_t(integrator, t) = tdir(integrator) * (t - integrator.t) < zero(integrator.t)
reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable) =
    integrator.t == tstop || (!stop_at_tstop && is_past_t(integrator, tstop))



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
    (; _dt, dtchangeable, tstops, subintegrators) = integrator

    # update step and dt before incrementing u; if dt is changeable and there is
    # a tstop within dt, reduce dt to tstop - t
    integrator.step += 1
    integrator.dt =
        !isempty(tstops) && dtchangeable ? tdir(integrator) * min(_dt, abs(first(tstops) - integrator.t)) :
        tdir(integrator) * _dt

    # Solve inner problems
    @timeit "Subintegrators" step_inner!(integrator)

    # increment t by dt, rounding to the first tstop if that is roughly
    # equivalent up to machine precision; the specific bound of 100 * eps...
    # is taken from OrdinaryDiffEq.jl
    t_plus_dt = integrator.t + integrator.dt
    t_unit = oneunit(integrator.t)
    max_t_error = 100 * eps(float(integrator.t / t_unit)) * t_unit
    integrator.tprev = integrator.t
    integrator.t = !isempty(tstops) && abs(first(tstops) - t_plus_dt) < max_t_error ? first(tstops) : t_plus_dt

    synchronize_subintegrators!(integrator)

    # remove tstops that were just reached
    while !isempty(tstops) && reached_tstop(integrator, first(tstops))
        pop!(tstops)
    end
end

# solvers need to define this interface
function step_inner!(integrator)
    step_inner!(integrator, integrator.cache)
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




# Lie-Trotter-Godunov Splitting Implementation
struct LieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
    # transfer_algs::TransferTupleType # Tuple of transfer algorithms from the master solution into the individual ones
end

struct LieTrotterGodunovCache{uType, tmpType, iiType}
    u::uType
    uprev::uType # True previous solution
    uprev2::tmpType # Previous solution used during time marching
    tmp::tmpType # Scratch
    inner_caches::iiType
end

# Dispatch for outer construction
function init_cache(prob::OperatorSplittingProblem, alg::LieTrotterGodunov; dt, kwargs...) # TODO
    @unpack f = prob
    @assert f isa GenericSplitFunction

    u = copy(prob.u0)
    uprev = copy(u)
    uprev2 = copy(u)
    tmp = copy(u)

    # Build inner integrator and connect solution vectors
    dof_ranges = f.dof_ranges
    # FIXME --->                                                                                 FIXME this deepcopy is to get rid of the supurious allocations
    inner_caches = ntuple(i->construct_inner_cache(get_operator(f, i), alg.inner_algs[i], view(u, deepcopy(dof_ranges[i])), view(uprev, deepcopy(dof_ranges[i]))), length(f.functions))
    return LieTrotterGodunovCache(u, uprev, uprev2, tmp, inner_caches)
end

function synchronize_subintegrators!(integrator::OperatorSplittingIntegrator)
    synchronize_subintegrator!(integrator.subintegrators, integrator)
end

@unroll function synchronize_subintegrator!(subintegrators::Tuple, integrator::OperatorSplittingIntegrator)
    @unroll for subintegrator in subintegrators
        synchronize_subintegrator!(subintegrator, integrator)
    end
end

function synchronize_subintegrator!(subintegrator::ThunderboltSubIntegrator, integrator::OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    subintegrator.t = t
    subintegrator.dt = dt
end

@inline function sync_inner!(subintegrator::ThunderboltSubIntegrator)
    subintegrator.uprev .= subintegrator.u
end

@unroll function sync_inner!(subintegrators::Tuple)
    @unroll for subintegrator in subintegrators
        sync_inner!(subintegrator)
    end
end

step_inner!(integrator::OperatorSplittingIntegrator, cache::LieTrotterGodunovCache) = step_inner!(integrator.subintegrators, cache)

@inline @unroll function step_inner!(subintegrators::Tuple, cache::LieTrotterGodunovCache)
    # We assume that the integrators are already synced
    @unpack u, uprev2, uprev, inner_caches = cache

    # Store current solution
    uprev .= u

    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        sync_inner!(subinteg)
        step_inner!(subinteg, inner_caches[i])
    end
end

# For testing purposes
struct ForwardEuler
end

mutable struct ForwardEulerCache{duType}
    du::duType
end

@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator, ::AbstractOperatorSplittingAlgorithm, cache)
    return (cache.tmp,)
end

# construct_inner_cache(f, alg, u::SubArray, uprev::SubArray) = error("Function type $typeof(f) not compatible with $alg!")
function construct_inner_cache(f::ODEFunction, alg::ForwardEuler, u::SubArray, uprev::SubArray)
    ForwardEulerCache(copy(uprev))
end

# Dispatch for recursive construction
function construct_inner_cache(f::AbstractOperatorSplitFunction, alg::LieTrotterGodunov, u::SubArray, uprev::SubArray)
    dof_ranges = f.dof_ranges
    # FIXME --->                                                                           FIXME this deepcopy is to get rid of the supurious allocations
    inner_caches = ntuple(i->construct_inner_cache(get_operator(f, i), alg.inner_algs[i], view(u, deepcopy(dof_ranges[i])), view(uprev, deepcopy(dof_ranges[i]))), length(dof_ranges))
    LieTrotterGodunovCache(u, uprev, copy(uprev), copy(u), inner_caches)
end

function step_inner!(integ, cache::ForwardEulerCache)
    @unpack f, dt, u, p, t = integ
    @unpack du = cache

    f(du, u, p, t)
    @. u += dt * du
end


function build_subintegrators_recursive(f::GenericSplitFunction, p::Tuple, cache::LieTrotterGodunovCache, u::AbstractArray, uprev::AbstractArray, t, dt)
    return ntuple(i -> 
        build_subintegrators_recursive(
            get_operator(f, i),
            p[i],
            cache.inner_caches[i],
            # FIXME this deepcopy is to get rid of the supurious allocations
            view(u, deepcopy(f.dof_ranges[i])),
            view(uprev, deepcopy(f.dof_ranges[i])),
            t, dt
        ), length(f.functions)
    )
end

function build_subintegrators_recursive(f::ODEFunction, p::Any, cache::Any, u::SubArray, uprev::SubArray, t, dt)
    return ThunderboltSubIntegrator(f, u, uprev, p, t, dt)
end

export ODEFunction, GenericSplitFunction, LieTrotterGodunov, ForwardEuler, DiffEqBase, OperatorSplittingProblem

end