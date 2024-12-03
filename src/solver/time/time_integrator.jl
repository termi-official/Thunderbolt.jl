mutable struct IntegratorStats
    iter::Int64
    nsuccess::Int64
    nreject::Int64
end

IntegratorStats() = IntegratorStats(0,0,0)

Base.@kwdef struct IntegratorOptions{tType}
    dtmin::tType = eps(Float64)
    dtmax::tType = tend-t0
end

"""
Internal helper to integrate a single inner operator
over some time interval.
"""
mutable struct ThunderboltTimeIntegrator{
    fType,
    uType,
    uprevType,
    indexSetType,
    tType,
    pType,
    cacheType,
    syncType,
    solType,
    heapType,
    tstopsType,
    saveatType,
    controllerType,
}  <: DiffEqBase.SciMLBase.DEIntegrator{#=alg_type=#Nothing, true, uType, tType} # FIXME alg
    const f::fType # Right hand side
    u::uType # Current local solution
    uprev::uprevType
    indexset::indexSetType
    p::pType
    t::tType
    tprev::tType
    dt::tType
    tdir::tType
    cache::cacheType
    synchronizer::syncType
    sol::solType
    const dtchangeable::Bool
    tstops::heapType
    _tstops::tstopsType # argument to __init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType # argument to __init used as default argument to reinit!
    controller::controllerType
    stats::IntegratorStats
    const opts::IntegratorOptions{tType}
end

# TimeChoiceIterator API
@inline function DiffEqBase.get_tmp_cache(integrator::ThunderboltTimeIntegrator)
    return (integrator.cache.tmp,)
end
@inline function DiffEqBase.get_tmp_cache(integrator::ThunderboltTimeIntegrator, alg::AbstractSolver, cache::AbstractTimeSolverCache)
    return (cache.tmp,)
end

# Interpolation
# TODO via https://github.com/SciML/SciMLBase.jl/blob/master/src/interpolation.jl
function (integrator::ThunderboltTimeIntegrator)(tmp, t)
    OS.linear_interpolation!(tmp, t, integrator.uprev, integrator.u, integrator.t-integrator.dt, integrator.t)
end

function DiffEqBase.isadaptive(integrator::ThunderboltTimeIntegrator)
    integrator.controller === nothing && return false
    if !DiffEqBase.isadaptive(integrator.alg)
        error("Algorithm $(integrator.alg) is not adaptive, but the integrator is trying to adapt. Aborting.")
    end
    return true
end


# Solution interface
should_accept_step(integrator::ThunderboltTimeIntegrator) = should_accept_step(integrator, integrator.cache, integrator.controller)
function should_accept_step(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    return true
end


# Controller interface
function reject_step!(integrator::ThunderboltTimeIntegrator)
    integrator.stats.nreject += 1
    reject_step!(integrator, integrator.cache, integrator.controller)
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache, controller)
    integrator.u .= integrator.uprev
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
end

adapt_dt!(integrator::ThunderboltTimeIntegrator) = adapt_dt!(integrator, integrator.cache, integrator.controller)
function adapt_dt!(integrator::ThunderboltTimeIntegrator, cache, controller)
    error("Step size control not implemented for $(alg).")
end
adapt_dt!(integrator::ThunderboltTimeIntegrator, cache, ::Nothing) = nothing

function accept_step!(integrator::ThunderboltTimeIntegrator)
    integrator.stats.nsuccess += 1
    accept_step!(integrator, integrator.cache, integrator.controller)
end
function accept_step!(integrator::ThunderboltTimeIntegrator, cache, controller)
    store_previous_info!(integrator)
end

function store_previous_info!(integrator::ThunderboltTimeIntegrator)
    if length(integrator.uprev) > 0 # Integrator can rollback
        update_uprev!(integrator)
    end
end

function step_header!(integrator::ThunderboltTimeIntegrator)
    # Accept or reject the step
    if integrator.stats.iter > 0
        if should_accept_step(integrator)
            accept_step!(integrator)
        else # Step should be rejected and hence repeated
            reject_step!(integrator)
        end
    end
    # Before stepping we might need to adjust the dt
    fix_dt_at_bounds!(integrator)
    fix_dt_at_tstops!(integrator)
    integrator.stats.iter += 1
end

function update_uprev!(integrator::ThunderboltTimeIntegrator)
    integrator.uprev .= integrator.u
end

function step_footer!(integrator::ThunderboltTimeIntegrator)
    if should_accept_step(integrator)
        integrator.tprev = integrator.t
        integrator.t += integrator.dt
        adapt_dt!(integrator) # Noop for non-adaptive algorithms
    end

    dtmin = 1e-12
    if integrator.dt < DiffEqBase.timedepentdtmin(integrator.t, integrator.opts.dtmin)
        error("dt too small ($(integrator.dt)). Aborting.")
    end
end

function handle_tstop!(integrator::ThunderboltTimeIntegrator)
    (; tstops) = integrator
    while !isempty(tstops) && OS.reached_tstop(integrator, first(tstops))
        pop!(tstops)
    end
end

function fix_dt_at_bounds!(integrator::ThunderboltTimeIntegrator)
    if integrator.tdir > 0
        integrator.dt = min(integrator.opts.dtmax, integrator.dt)
    else
        integrator.dt = max(integrator.opts.dtmax, integrator.dt)
    end
    dtmin = DiffEqBase.timedepentdtmin(integrator.t, integrator.opts.dtmin)
    if integrator.tdir > 0
        integrator.dt = max(integrator.dt, dtmin)
    else
        integrator.dt = min(integrator.dt, dtmin)
    end
    return nothing
end

function DiffEqBase.step!(integrator::ThunderboltTimeIntegrator, dt, stop_at_tdt = false)
    dt <= zero(dt) && error("dt must be positive")
    while !OS.reached_tstop(integrator,  integrator.t, stop_at_tdt)
        step_header!(integrator)
        perform_step!(integrator, integrator.cache) #|| error("Time integration failed at t=$(integrator.t).") # remove this
        step_footer!(integrator)
    end

    handle_tstop!(integrator)
end
function OS.synchronize_subintegrator!(subintegrator::ThunderboltTimeIntegrator, integrator::OS.OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    subintegrator.t = t
    subintegrator.dt = dt
end
# TODO some operator splitting methods require to go back in time, so we need to figure out what the best way is.
OS.tdir(::ThunderboltTimeIntegrator) = 1

# TODO Any -> cache supertype
function OS.advance_solution_to!(integrator::ThunderboltTimeIntegrator, cache::Any, tend; kwargs...)
    @unpack f, t = integrator
    dt = tend-t
    dt ≈ 0.0 || DiffEqBase.step!(integrator, dt, true)
end
@inline function OS.prepare_local_step!(uparent, subintegrator::ThunderboltTimeIntegrator)
    # Copy solution into subproblem
    uparentview      = @view uparent[subintegrator.indexset]
    subintegrator.u .= uparentview
    # Mark previous solution, if necessary
    if subintegrator.uprev !== nothing && length(subintegrator.uprev) > 0
        subintegrator.uprev .= subintegrator.u
    end
    syncronize_parameters!(subintegrator, subintegrator.f, subintegrator.synchronizer)
end
@inline function OS.finalize_local_step!(uparent, subintegrator::ThunderboltTimeIntegrator)
    # Copy solution out of subproblem
    #
    uparentview = @view uparent[subintegrator.indexset]
    uparentview .= subintegrator.u
end
# Glue code
function OS.build_subintegrators_recursive(f, synchronizer, p::Any, cache::AbstractTimeSolverCache, t::tType, dt::tType, dof_range, uparent, tstops, _tstops, saveat, _saveat) where tType
    integrator = Thunderbolt.ThunderboltTimeIntegrator(
        f,
        cache.uₙ,
        cache.uₙ₋₁,
        dof_range,
        p,
        t,
        t,
        dt,
        tType(sign(dt)), #tdir,
        cache,
        synchronizer,
        nothing, # FIXME sol
        true, #dtchangeable
        tstops,
        _tstops,
        saveat,
        _saveat,
        nothing, # FIXME controller
        IntegratorStats(),
        IntegratorOptions(
            dtmin = eps(tType)
            dtmax = tType(Inf)
        ),
    )
    # This makes sure that the parameters are set correctly for the first time step
    syncronize_parameters!(integrator, f, synchronizer)
    return integrator
end
function OS.construct_inner_cache(f, alg::AbstractSolver; u0, t0, kwargs...)
    return Thunderbolt.setup_solver_cache(f, alg, t0)
end
OS.recursive_null_parameters(stuff::Union{AbstractSemidiscreteProblem, AbstractSemidiscreteFunction}) = OS.DiffEqBase.NullParameters()
syncronize_parameters!(integ, f, ::OS.NoExternalSynchronization) = nothing

function DiffEqBase.__init(
    prob::AbstractSemidiscreteProblem,
    alg::AbstractSolver,
    args...;
    dt,
    tstops = (),
    saveat = nothing,
    save_everystep = false,
    callback = nothing,
    advance_to_tstop = false,
    adaptive = false,
    controller = nothing,
    save_func = (u, t) -> copy(u),                  # custom kwarg
    dtchangeable = true,                            # custom kwarg
    syncronizer = OS.NoExternalSynchronization(),   # custom kwarg
    kwargs...,
)
    (; f, u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    dt = tf > t0 ? dt : -dt

    _tstops = tstops
    _saveat = saveat
    tstops, saveat = OS.tstops_and_saveat_heaps(t0, tf, tstops, saveat)

    sol = DiffEqBase.build_solution(prob, alg, typeof(t0)[], typeof(save_func(u0, t0))[])

    callback = DiffEqBase.CallbackSet(callback)

    cache = init_cache(prob, alg; t0)

    cache.uₙ   .= u0

    if controller === nothing && adaptive
        controller = default_controller(alg, cache)
    end

    integrator = ThunderboltTimeIntegrator(
        f,
        cache.uₙ,
        cache.uₙ₋₁,
        1:length(u0),
        p,
        t0,
        t0,
        dt,
        tType(sign(dt)), #tdir,
        cache,
        syncronizer,
        sol,
        true,
        tstops,
        _tstops,
        saveat,
        _saveat,
        adaptive ? controller : nothing,
        IntegratorStats(),
        IntegratorOptions(
            dtmin = eps(tType)
            dtmax = tType(tf-t0)
        )
    )
    # DiffEqBase.initialize!(callback, u0, t0, integrator) # Do I need this?
    return integrator
end

function DiffEqBase.solve!(integrator::ThunderboltTimeIntegrator)
    while !isempty(integrator.tstops)
        OS.advance_solution_to!(integrator, integrator.cache, first(integrator.tstops))
    end
    # DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, DiffEqBase.ReturnCode.Success)
    return integrator.sol
end

# Some common getters
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int) = get_parent_index(integ, local_idx, integ.indexset)
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int, indexset::AbstractVector) = indexset[local_idx]
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int, range::AbstractUnitRange) = first(range) + local_idx - 1
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int, range::StepRange) = first(range) + range.step*(local_idx - 1)

# Compat with OrdinaryDiffEq
function perform_step!(integ::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache)
    if !perform_step!(integ.f, cache, integ.t, integ.dt)
        if integ.sol !== nothing # FIXME
            integ.sol = DiffEqBase.solution_new_retcode(integ.sol, DiffEqBase.ReturnCode.Failure)
        end
        return false
    end
    return true
end

function init_cache(prob, alg; t0, kwargs...)
    return setup_solver_cache(prob.f, alg, t0)
end
