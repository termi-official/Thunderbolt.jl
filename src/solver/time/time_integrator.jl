mutable struct IntegratorStats
    iter::Int64
    nsuccess::Int64
    nreject::Int64
end

IntegratorStats() = IntegratorStats(0,0,0)

Base.@kwdef struct IntegratorOptions{tType}
    dtmin::tType = eps(tType)
    dtmax::tType = Inf
    failfactor::tType = 0.25
    verbose::Bool = false
end

"""
Internal helper to integrate a single inner operator
over some time interval.
"""
mutable struct ThunderboltTimeIntegrator{
    algType,
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
}  <: DiffEqBase.SciMLBase.DEIntegrator{algType, true, uType, tType} # FIXME alg
    alg::algType
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
    force_stepfail::Bool
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
function should_accept_step(integrator::ThunderboltTimeIntegrator)
    if integrator.force_stepfail
        return false
    end
    return should_accept_step(integrator, integrator.cache, integrator.controller)
end
function should_accept_step(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    return !(integrator.force_stepfail)
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
    modify_dt_for_tstops!(integrator)
    integrator.stats.iter += 1
    integrator.force_stepfail = false
end

function update_uprev!(integrator::ThunderboltTimeIntegrator)
    integrator.uprev .= integrator.u
end

function step_footer!(integrator::ThunderboltTimeIntegrator)
    if should_accept_step(integrator)
        integrator.tprev = integrator.t
        integrator.t += integrator.dt
        adapt_dt!(integrator) # Noop for non-adaptive algorithms
    elseif integrator.force_stepfail
        if integrator.dtchangeable
            integrator.dt *= integrator.opts.failfactor
        else
            error("Integration over [$(integrator.t), $(integrator.t+integrator.dt)] failed and cannot change dt to recover. Aborting.")
        end
    end

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

function modify_dt_for_tstops!(integrator::ThunderboltTimeIntegrator)
    if DiffEqBase.has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = DiffEqBase.first_tstop(integrator)
        if DiffEqBase.isadaptive(integrator)
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif iszero(integrator.dtcache) && integrator.dtchangeable
            integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
        elseif integrator.dtchangeable && !integrator.force_stepfail
            # always try to step! with dtcache, but lower if a tstop
            # however, if force_stepfail then don't set to dtcache, and no tstop worry
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dtcache), abs(tdir_tstop - tdir_t)) # step! to the end
        end
    end
end

function DiffEqBase.step!(integrator::ThunderboltTimeIntegrator, dt, stop_at_tdt = false)
    dt <= zero(dt) && error("dt must be positive")
    tstop = integrator.t + dt
    while !OS.reached_tstop(integrator, tstop, stop_at_tdt)
        step_header!(integrator)
        if !perform_step!(integrator, integrator.cache)
            integrator.force_stepfail = true
        end
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
        nothing, # FIXME
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
            dtmin = eps(tType),
            dtmax = tType(Inf),
        ),
        false,
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
    verbose = false,
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
    tType = typeof(dt)

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
        alg,
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
            dtmin = eps(tType),
            dtmax = tType(tf-t0),
            verbose = verbose,
        ),
        false
    )
    # DiffEqBase.initialize!(callback, u0, t0, integrator) # Do I need this?
    return integrator
end

function DiffEqBase.first_tstop(integrator::ThunderboltTimeIntegrator)
    return first(integrator.tstops)
end

function DiffEqBase.has_tstop(integrator::ThunderboltTimeIntegrator)
    return length(integrator.tstops) > 0
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
    integ.opts.verbose && @info "Time integration on [$(integ.t), $(integ.t+integ.dt)] (Δt=$(integ.dt))"
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
