mutable struct IntegratorStats
    naccept::Int64
    nreject::Int64
    # TODO inner solver stats
end

IntegratorStats() = IntegratorStats(0,0)

Base.@kwdef  struct IntegratorOptions{tType, #=msgType,=# F1, F2, F3, F4, F5, SType, tstopsType, saveatType, discType, tcache, savecache, disccache}
    dtmin::tType = eps(tType)
    dtmax::tType = Inf
    failfactor::tType = 4.0
    verbose::Bool = false
    adaptive::Bool = false # Redundant with the dispatch on SciMLBase.isadaptive below (alg adaptive + controller not nothing)
    maxiters::Int = 1000000
    # OrdinaryDiffEqCore compat
    force_dtmin::Bool = false
    progress::Bool = false # FIXME
    # progress_steps::Int = 0
    # progress_name::String = ""
    # progress_message::msgType = ""
    # progress_id::Symbol = :msg
    internalnorm::F1 = DiffEqBase.ODE_DEFAULT_NORM
    internalopnorm::F2 = LinearAlgebra.opnorm
    isoutofdomain::F3 = DiffEqBase.ODE_DEFAULT_ISOUTOFDOMAIN
    unstable_check::F4 = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK
    save_idxs::SType = nothing
    tstops::tstopsType = nothing
    saveat::saveatType = nothing
    save_end::Bool = true
    d_discontinuities::discType = nothing
    tstops_cache::tcache = ()
    saveat_cache::savecache = ()
    d_discontinuities_cache::disccache = ()
    callback::F5
    dense::Bool = false
    save_on::Bool = false
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
    callbackcacheType,
    syncType,
    solType,
    controllerType,
}  <: SciMLBase.DEIntegrator{algType, true, uType, tType}
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
    callback_cache::callbackcacheType
    synchronizer::syncType
    sol::solType
    const dtchangeable::Bool
    controller::controllerType
    stats::IntegratorStats
    const opts::IntegratorOptions{tType}
    # OrdinaryDiffEqCore compat
    force_stepfail::Bool # This is a flag for the inner solver to communicate that it failed
    iter::Int64 # TODO move into stats
    u_modified::Bool
    isout::Bool
    reeval_fsal::Bool
    last_step_failed::Bool
    saveiter::Int
    saveiter_dense::Int
    just_hit_tstop::Bool
end

# ----------------------------------- SciMLBase.jl Interface ------------------------------------
SciMLBase.has_stats(::ThunderboltTimeIntegrator) = true

SciMLBase.has_tstop(integrator::ThunderboltTimeIntegrator) = !isempty(integrator.opts.tstops)
SciMLBase.first_tstop(integrator::ThunderboltTimeIntegrator) = first(integrator.opts.tstops)
SciMLBase.pop_tstop!(integrator::ThunderboltTimeIntegrator) = pop!(integrator.opts.tstops)

@inline function SciMLBase.get_tmp_cache(integrator::ThunderboltTimeIntegrator)
    return (integrator.cache.tmp,)
end
@inline function SciMLBase.get_tmp_cache(integrator::ThunderboltTimeIntegrator, alg::AbstractSolver, cache::AbstractTimeSolverCache)
    return (cache.tmp,)
end

function SciMLBase.terminate!(integrator::ThunderboltTimeIntegrator, retcode = ReturnCode.Terminated)
    integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, retcode)
    integrator.opts.tstops.valtree = typeof(integrator.opts.tstops.valtree)()
end

# @inline function SciMLBase.get_du(integrator::ThunderboltTimeIntegrator)
# end

@inline SciMLBase.get_proposed_dt(integrator::ThunderboltTimeIntegrator) = integrator.dt

@inline function SciMLBase.u_modified!(integrator::ThunderboltTimeIntegrator, bool::Bool)
    integrator.u_modified = bool
end

SciMLBase.get_sol(integrator::ThunderboltTimeIntegrator) = integrator.sol

# Interpolation
# TODO via https://github.com/SciML/SciMLBase.jl/blob/master/src/interpolation.jl
function (integrator::ThunderboltTimeIntegrator)(tmp, t)
    OS.linear_interpolation!(tmp, t, integrator.uprev, integrator.u, integrator.t-integrator.dt, integrator.t)
end

function SciMLBase.isadaptive(integrator::ThunderboltTimeIntegrator)
    integrator.controller === nothing && return false
    if !SciMLBase.isadaptive(integrator.alg)
        error("Algorithm $(integrator.alg) is not adaptive, but the integrator is trying to adapt. Aborting.")
    end
    return true
end

function SciMLBase.last_step_failed(integrator::ThunderboltTimeIntegrator)
    integrator.last_step_failed
end

SciMLBase.postamble!(integrator::ThunderboltTimeIntegrator) = _postamble!(integrator)

function SciMLBase.savevalues!(integrator::ThunderboltTimeIntegrator, force_save = false, reduce_size = true)
    OrdinaryDiffEqCore._savevalues!(integrator, force_save, reduce_size)
end

# ----------------------------------- DiffEqBase.jl Interface ------------------------------------
DiffEqBase.get_tstops(integ::ThunderboltTimeIntegrator) = integ.opts.tstops
DiffEqBase.get_tstops_array(integ::ThunderboltTimeIntegrator) = get_tstops(integ).valtree
DiffEqBase.get_tstops_max(integ::ThunderboltTimeIntegrator) = maximum(get_tstops_array(integ))


# ----------------------------------- OrdinaryDiffEqCore compat ----------------------------------
function _postamble!(integrator)
    DiffEqBase.finalize!(integrator.opts.callback, integrator.u, integrator.t, integrator)
    OrdinaryDiffEqCore.solution_endpoint_match_cur_integrator!(integrator)
    fix_solution_buffer_sizes!(integrator, integrator.sol)
    finalize_integration_monitor(integrator)
end

OrdinaryDiffEqCore.alg_extrapolates(alg::AbstractSolver) = false
OrdinaryDiffEqCore.alg_extrapolates(::Nothing) = false # HOTFIX REMOVE THIS

# --------------------------- New Interface Stuff (to be upstreamed) ---------------------------------

# Solution looping interface
function should_accept_step(integrator::ThunderboltTimeIntegrator)
    if integrator.force_stepfail || integrator.isout
        return false
    end
    return should_accept_step(integrator, integrator.cache, integrator.controller)
end
function should_accept_step(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    return !(integrator.force_stepfail)
end

function accept_step!(integrator::ThunderboltTimeIntegrator)
    OrdinaryDiffEqCore.increment_accept!(integrator.stats)
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
    if !is_first_iteration(integrator)
        if should_accept_step(integrator)
            accept_step!(integrator)
        else # Step should be rejected and hence repeated
            reject_step!(integrator)
        end
    elseif integrator.u_modified # && integrator.iter == 0
        update_uprev!(integrator)
    end

    # Before stepping we might need to adjust the dt
    increment_iteration(integrator)
    OrdinaryDiffEqCore.choose_algorithm!(integrator, integrator.cache)
    OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator)
    OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator)
    integrator.force_stepfail = false
end

function update_uprev!(integrator::ThunderboltTimeIntegrator)
    # # OrdinaryDiffEqCore.update_uprev!(integrator) # FIXME recover
    # if alg_extrapolates(integrator.alg)
    #     if isinplace(integrator.sol.prob)
    #         SciMLBase.recursivecopy!(integrator.uprev2, integrator.uprev)
    #     else
    #         integrator.uprev2 = integrator.uprev
    #     end
    # end
    # if isinplace(integrator.sol.prob) # This should be dispatched in the integrator directly
        SciMLBase.recursivecopy!(integrator.uprev, integrator.u)
        if integrator.alg isa OrdinaryDiffEqCore.DAEAlgorithm
            SciMLBase.recursivecopy!(integrator.duprev, integrator.du)
        end
    # else
    #     integrator.uprev = integrator.u
    #     if integrator.alg isa DAEAlgorithm
    #         integrator.duprev = integrator.du
    #     end
    # end
    nothing
end

function controller_message_on_dtmin_error(integrator::SciMLBase.DEIntegrator)
    if isdefined(integrator, :EEst)
       return ", and step error estimate = $(integrator.EEst)"
    else
        return ""
    end
end

function SciMLBase.check_error(integrator::ThunderboltTimeIntegrator)
    if integrator.sol.retcode ∉ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default)
        return integrator.sol.retcode
    end
    opts = integrator.opts
    verbose = opts.verbose
    # This implementation is intended to be used for ODEIntegrator and
    # SDEIntegrator.
    if isnan(integrator.dt)
        if verbose
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        end
        return SciMLBase.ReturnCode.DtNaN
    end
    if hasproperty(integrator, :iter) && hasproperty(opts, :maxiters) && integrator.iter > opts.maxiters
        if verbose
            @warn("Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).")
        end
        return SciMLBase.ReturnCode.MaxIters
    end

    # The last part:
    # Bail out if we take a step with dt less than the minimum value (which may be time dependent)
    # except if we are successfully taking such a small timestep is to hit a tstop exactly
    # We also exit if the ODE is unstable according to a user chosen callback
    # but only if we accepted the step to prevent from bailing out as unstable
    # when we just took way too big a step)
    step_accepted = should_accept_step(integrator)
    step_rejected = !step_accepted
    force_dtmin   = hasproperty(integrator, :force_dtmin) && integrator.force_dtmin
    if !force_dtmin && SciMLBase.isadaptive(integrator)
        dt_below_min      = abs(integrator.dt) ≤ abs(opts.dtmin)
        before_next_tstop = SciMLBase.has_tstop(integrator) ? integrator.t + integrator.dt < integrator.tdir * SciMLBase.first_tstop(integrator) : true
        if dt_below_min && (step_rejected || before_next_tstop)
            if verbose
                controller_string = controller_message_on_dtmin_error(integrator)
                @warn("dt($(integrator.dt)) <= dtmin($(opts.dtmin)) at t=$(integrator.t)$(controller_string). Aborting. There is either an error in your model specification or the true solution is unstable.")
            end
            return SciMLBase.ReturnCode.DtLessThanMin
        elseif step_rejected && integrator.t isa AbstractFloat &&
               abs(integrator.dt) <= abs(eps(integrator.t)) # = DiffEqBase.timedepentdtmin(integrator)
            if verbose
                controller_string = controller_message_on_dtmin_error(integrator)
                @warn("At t=$(integrator.t), dt was forced below floating point epsilon $(integrator.dt)$(controller_string). Aborting. There is either an error in your model specification or the true solution is unstable (or the true solution can not be represented in the precision of $(eltype(integrator.u))).")
            end
            return SciMLBase.ReturnCode.Unstable
        end
    end
    if step_accepted && (hasproperty(opts, :unstable_check) && 
       opts.unstable_check(integrator.dt, integrator.u, integrator.p, integrator.t))
        if verbose
            @warn("Instability detected. Aborting")
        end
        return SciMLBase.ReturnCode.Unstable
    end
    if SciMLBase.last_step_failed(integrator) && !SciMLBase.isadaptive(integrator)
        if verbose
            @warn("Newton steps could not converge and algorithm is not adaptive. Use a lower dt.")
        end
        return SciMLBase.ReturnCode.ConvergenceFailure
    end
    return SciMLBase.ReturnCode.Success
end

function footer_reset_flags!(integrator)
    integrator.u_modified = false
end

function fix_solution_buffer_sizes!(integrator, sol)
    resize!(integrator.sol.t, integrator.saveiter)
    resize!(integrator.sol.u, integrator.saveiter)
    if !(integrator.sol isa SciMLBase.DAESolution)
        resize!(integrator.sol.k, integrator.saveiter_dense)
    end
end

function setup_validity_flags!(integrator, t_next)
    integrator.isout = integrator.opts.isoutofdomain(integrator.u, integrator.p, t_next)
end

function step_footer!(integrator::ThunderboltTimeIntegrator)
    ttmp = integrator.t + integrator.tdir * integrator.dt

    footer_reset_flags!(integrator)
    setup_validity_flags!(integrator, ttmp)

    if should_accept_step(integrator)
        integrator.last_step_failed = false
        integrator.tprev = integrator.t
        integrator.t = OrdinaryDiffEqCore.fixed_t_for_floatingpoint_error!(integrator, ttmp)
        OrdinaryDiffEqCore.handle_callbacks!(integrator)
        adapt_dt!(integrator) # Noop for non-adaptive algorithms
    elseif integrator.force_stepfail
        if SciMLBase.isadaptive(integrator)
            OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
        # elseif integrator.dtchangeable # Non-adaptive but can change dt
        #     integrator.dt *= integrator.opts.failfactor
        elseif integrator.last_step_failed
            return
        end
        integrator.last_step_failed = true
    end

    integration_monitor_step(integrator)

    nothing
end

notify_integrator_hit_tstop!(integrator) = integrator.just_hit_tstop = true

is_first_iteration(integrator) = integrator.iter == 0
increment_iteration(integrator) = integrator.iter += 1

function integration_monitor_step(integrator)
    if integrator.opts.progress && integrator.iter % integrator.opts.progress_steps == 0
        OrdinaryDiffEqCore.log_step!(integrator.opts.progress_name, integrator.opts.progress_id,
            integrator.opts.progress_message, integrator.dt, integrator.u,
            integrator.p, integrator.t, integrator.sol.prob.tspan)
    end
end

function finalize_integration_monitor(integrator)
    if integrator.opts.progress
        @logmsg(LogLevel(-1),
            integrator.opts.progress_name,
            _id=integrator.opts.progress_id,
            message=integrator.opts.progress_message(integrator.dt, integrator.u,
                integrator.p, integrator.t),
            progress="done")
    end
end

notify_integrator_hit_tstop!(integrator::ThunderboltTimeIntegrator) = nothing

# TODO upstream into OrdinaryDiffEqCore
function compute_rate_prototype(prob)
    u = prob.u0

    tType = eltype(prob.tspan)
    tTypeNoUnits = typeof(one(tType))

    isdae = (
        prob.f.mass_matrix != I &&
        !(prob.f.mass_matrix isa Tuple) &&
        ArrayInterface.issingular(prob.f.mass_matrix)
    )
    if !isdae && isinplace(prob) && u isa AbstractArray && eltype(u) <: Number &&
        uBottomEltypeNoUnits == uBottomEltype && tType == tTypeNoUnits # Could this be more efficient for other arrays?
        return SciMLBase.recursivecopy(u)
    else
        _compute_rate_prototype_mass_matrix_form(prob)
    end
end
function compute_rate_prototype(prob::SciMLBase.DiscreteProblem)
    _compute_rate_prototype_mass_matrix_form(prob)
end
function _compute_rate_prototype_mass_matrix_form(prob)
    u = prob.u0

    tType = eltype(prob.tspan)
    tTypeNoUnits = typeof(one(tType))
    uBottomEltype        = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)
    if (uBottomEltypeNoUnits == uBottomEltype && tType == tTypeNoUnits) || eltype(u) <: Enum
        return u
    else # has units!
        return u / oneunit(tType)
    end
end
function compute_rate_prototype(prob::SciMLBase.DAEProblem)
    return prob.du0
end

function compute_rate_prototype(prob::AbstractSemidiscreteProblem)
    _compute_rate_prototype_mass_matrix_form(prob)
end

# Slightly modified functions are here
function OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator::ThunderboltTimeIntegrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        if integrator.opts.adaptive
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif integrator.dtchangeable
            dtpropose = SciMLBase.get_proposed_dt(integrator)
            if iszero(dtpropose)
                integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
            elseif !integrator.force_stepfail
                # always try to step! with dtcache, but lower if a tstop
                # however, if force_stepfail then don't set to dtcache, and no tstop worry
                integrator.dt = integrator.tdir *
                                min(abs(dtpropose), abs(tdir_tstop - tdir_t)) # step! to the end
            end
        end
    end
end

function OrdinaryDiffEqCore.handle_tstop!(integrator::ThunderboltTimeIntegrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        if tdir_t == tdir_tstop
            while tdir_t == tdir_tstop #remove all redundant copies
                res = SciMLBase.pop_tstop!(integrator)
                SciMLBase.has_tstop(integrator) ? (tdir_tstop = SciMLBase.first_tstop(integrator)) : break
            end
            notify_integrator_hit_tstop!(integrator)
        elseif tdir_t > tdir_tstop
            if !integrator.dtchangeable
                SciMLBase.change_t_via_interpolation!(integrator,
                    integrator.tdir *
                    SciMLBase.pop_tstop!(integrator), Val{true})
                    notify_integrator_hit_tstop!(integrator)
            else
                error("Something went wrong. Integrator stepped past tstops but the algorithm was dtchangeable. Please report this error.")
            end
        end
    end
    return nothing
end

@inline function SciMLBase.step!(integrator::ThunderboltTimeIntegrator)
    @inbounds while true # Emulate do-while
        step_header!(integrator)
        if SciMLBase.check_error!(integrator) != SciMLBase.ReturnCode.Success
            return
        end
        OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
        step_footer!(integrator)
        # Exit condition
        should_accept_step(integrator) && break
    end
    @inbounds OrdinaryDiffEqCore.handle_tstop!(integrator)
end

# Controller interface
function reject_step!(integrator::ThunderboltTimeIntegrator)
    OrdinaryDiffEqCore.increment_reject!(integrator.stats)
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

function SciMLBase.__init(
    prob::AbstractSemidiscreteProblem,
    alg::AbstractSolver,
    args...;
    dt,
    saveat = (),
    tstops = (),
    d_discontinuities = (),
    ts_init = (),
    ks_init = (),
    save_end = nothing,
    save_everystep = false,
    save_idxs = nothing,
    callback = nothing,
    advance_to_tstop = false,
    adaptive = false,
    verbose = false,
    alias_u0 = true,
    # alias_du0 = false,
    controller = nothing,
    maxiters = 1000000,
    dense = save_everystep &&
                    !(alg isa DAEAlgorithm) && !(prob isa DiscreteProblem) &&
                    isempty(saveat),
    dtmin = zero(tType),
    dtmax = tType(tf-t0),
    syncronizer = OS.NoExternalSynchronization(),   # custom kwarg
    kwargs...,
)
    (; f, u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    tdir = tf > t0 ? 1.0 : -1.0
    tType = typeof(dt)

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

    save_end = save_end === nothing ? save_everystep || isempty(saveat) || saveat isa Number || tf in saveat : save_end

    # Setup solution buffers
    u  = setup_u(prob, alg, alias_u0)
    uType                = typeof(u)
    uBottomEltype        = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)

    # Setup callbacks
    callbacks_internal = SciMLBase.CallbackSet(callback)
    max_len_cb = DiffEqBase.max_vector_callback_length_int(callbacks_internal)
    if max_len_cb !== nothing
        uBottomEltypeReal = real(uBottomEltype)
        if SciMLBase.isinplace(prob)
            callback_cache = SciMLBase.CallbackCache(u, max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        else
            callback_cache = SciMLBase.CallbackCache(max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        end
    else
        callback_cache = nothing
    end

    # Setup solution
    save_idxs, saved_subsystem = SciMLBase.get_save_idxs_and_saved_subsystem(prob, save_idxs)

    rate_prototype = compute_rate_prototype(prob)
    rateType = typeof(rate_prototype)
    if save_idxs === nothing
        ksEltype = Vector{rateType}
    else
        ks_prototype = rate_prototype[save_idxs]
        ksEltype = Vector{typeof(ks_prototype)}
    end

    ts = ts_init === () ? tType[] : convert(Vector{tType}, ts_init)
    ks = ks_init === () ? ksEltype[] : convert(Vector{ksEltype}, ks_init)

    sol = SciMLBase.build_solution(
        prob, alg, ts, uType[],
        dense = dense, k = ks, saved_subsystem = saved_subsystem,
        calculate_error = false
    )

    # Setup algorithm cache
    cache = init_cache(prob, alg; dt = dt, u = u)

    # Setup controller
    if controller === nothing && adaptive
        controller = default_controller(alg, cache)
    end

    # Setup the actual integrator object
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
        tdir,
        cache,
        callback_cache,
        syncronizer,
        sol,
        true,
        adaptive ? controller : nothing,
        IntegratorStats(),
        IntegratorOptions(
            dtmin = dtmin,
            dtmax = dtmax,
            verbose = verbose,
            adaptive = adaptive,
            maxiters = maxiters,
            callback = callbacks_internal,
            save_end = save_end,
            tstops = tstops_internal,
            saveat = saveat_internal,
            d_discontinuities = d_discontinuities_internal,
            tstops_cache = tstops,
            saveat_cache = saveat,
            d_discontinuities_cache = d_discontinuities,
        ),
        false,
        0,
        false,
        false,
        false,
        false,
        0,
        0,
        false,
    )
    OrdinaryDiffEqCore.initialize_callbacks!(integrator)
    DiffEqBase.initialize!(integrator, integrator.cache)
    # FIXME
    cache.uₙ .= u
    cache.uₙ₋₁ .= u

    if _tstops !== nothing
        tstops = _tstops(parameter_values(integrator), prob.tspan)
        for tstop in tstops
            add_tstop!(integrator, tstop)
        end
    end

    OrdinaryDiffEqCore.handle_dt!(integrator)

    return integrator
end

# TODO what exactly should I do here :)
DiffEqBase.initialize!(integrator::ThunderboltTimeIntegrator, ::AbstractTimeSolverCache) = nothing

function SciMLBase.solve!(integrator::ThunderboltTimeIntegrator)
    @inbounds while SciMLBase.has_tstop(integrator)
        while integrator.tdir * integrator.t < SciMLBase.SciMLBase.first_tstop(integrator)
            step_header!(integrator)
            if SciMLBase.check_error!(integrator) != SciMLBase.ReturnCode.Success
                return integrator.sol
            end
            OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
            step_footer!(integrator)
            if !SciMLBase.has_tstop(integrator)
                break
            end
        end
        OrdinaryDiffEqCore.handle_tstop!(integrator)
    end
    OrdinaryDiffEqCore.postamble!(integrator)
    if integrator.sol.retcode != SciMLBase.ReturnCode.Default
        return integrator.sol
    end
    return integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Success)
end

# Compat with OrdinaryDiffEq
function OrdinaryDiffEqCore.perform_step!(integ::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache)
    integ.opts.verbose && @info "Time integration on [$(integ.t), $(integ.t+integ.dt)] (Δt=$(integ.dt))"
    if !perform_step!(integ.f, cache, integ.t, integ.dt)
        integ.force_stepfail = true
    end
    return nothing
end

function init_cache(prob, alg; dt, kwargs...)
    return setup_solver_cache(prob.f, alg, prob.tspan[1])
end

function setup_u(prob::AbstractSemidiscreteProblem, solver, alias_u0)
    if alias_u0
        return prob.u0
    else
        return SciMLBase.recursivecopy(prob.u0)
    end
end

OrdinaryDiffEqCore.choose_algorithm!(integrator, cache::AbstractTimeSolverCache) = nothing


# -------------------------- Operator Splitting Compat ------------------------
function OS.synchronize_subintegrator!(subintegrator::ThunderboltTimeIntegrator, integrator::OS.OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    subintegrator.t = t
    subintegrator.dt = dt
end
# TODO some operator splitting methods require to go back in time, so we need to figure out what the best way is.
OS.tdir(integ::ThunderboltTimeIntegrator) = integ.tdir

function OS.advance_solution_to!(integrator::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache, tend; kwargs...)
    # @unpack f, t = integrator
    # dt = tend-t
    # dt ≈ 0.0 || SciMLBase.step!(integrator, dt, true)
    @inbounds while integrator.tdir * integrator.t < tend
        integrator.dt ≈ 0.0 && error("???")
        @info integrator.t, integrator.dt, tend
        step_header!(integrator)
        if OrdinaryDiffEqCore.check_error!(integrator) != SciMLBase.ReturnCode.Success
            return
        end
        OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
        step_footer!(integrator)
    end
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

struct DummyODESolution <: SciMLBase.AbstractODESolution{Float64, 2, Vector{Float64}}
    retcode::SciMLBase.ReturnCode.T
end
DummyODESolution() = DummyODESolution(SciMLBase.ReturnCode.Default)
function SciMLBase.solution_new_retcode(sol::DummyODESolution, retcode)
    return DiffEqBase.@set sol.retcode = retcode
end
fix_solution_buffer_sizes!(integrator, sol::DummyODESolution) = nothing

OS.recursive_null_parameters(stuff::Union{AbstractSemidiscreteProblem, AbstractSemidiscreteFunction}) = SciMLBase.NullParameters()
syncronize_parameters!(integ, f, ::OS.NoExternalSynchronization) = nothing

function OS.build_subintegrators_with_cache(
    f::AbstractSemidiscreteFunction, alg::AbstractSolver, p,
    uprevouter::AbstractVector, uouter::AbstractVector,
    solution_indices,
    t0, dt, tf,
    tstops, saveat, d_discontinuities, callback,
    adaptive, verbose,
    save_end = false,
    controller = nothing,
)
    uprev = @view uprevouter[solution_indices]
    u     = @view uouter[solution_indices]

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    tdir = tf > t0 ? 1.0 : -1.0
    tType = typeof(dt)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, (t0, tf))
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, (t0, tf))
    d_discontinuities_internal = OrdinaryDiffEqCore.initialize_d_discontinuities(tType, d_discontinuities, (t0, tf))

    cache = setup_solver_cache(f, alg, t0; uprev=uprev, u=u)

    # Setup solution buffers
    uType                = typeof(u)
    uBottomEltype        = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)

    # Setup callbacks
    callbacks_internal = SciMLBase.CallbackSet(callback)
    max_len_cb = DiffEqBase.max_vector_callback_length_int(callbacks_internal)
    if max_len_cb !== nothing
        uBottomEltypeReal = real(uBottomEltype)
        # if SciMLBase.isinplace(prob)
        #     callback_cache = SciMLBase.CallbackCache(u, max_len_cb, uBottomEltypeReal,
        #         uBottomEltypeReal)
        # else
            callback_cache = SciMLBase.CallbackCache(max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        # end
    else
        callback_cache = nothing
    end

    # # Setup solution
    # save_idxs, saved_subsystem = SciMLBase.get_save_idxs_and_saved_subsystem(prob, save_idxs)

    # rate_prototype = compute_rate_prototype(prob)
    # rateType = typeof(rate_prototype)
    # if save_idxs === nothing
    #     ksEltype = Vector{rateType}
    # else
    #     ks_prototype = rate_prototype[save_idxs]
    #     ksEltype = Vector{typeof(ks_prototype)}
    # end

    ts = tType[]
    ks = uType[]

    # sol = SciMLBase.build_solution(
    #     prob, alg, ts, uType[],
    #     # dense = dense, k = ks, saved_subsystem = saved_subsystem,
    #     calculate_error = false
    # )
    sol = DummyODESolution()

    if controller === nothing && adaptive
        controller = default_controller(alg, cache)
    end

    save_end = save_end === nothing ? save_everystep || isempty(saveat) || saveat isa Number || tf in saveat : save_end

    # Setup the actual integrator object
    integrator = ThunderboltTimeIntegrator(
        alg,
        f,
        cache.uₙ,
        cache.uₙ₋₁,
        1:length(u),
        p,
        t0,
        t0,
        dt,
        tdir,
        cache,
        callback_cache,
        OS.NoExternalSynchronization(),
        sol,
        true,
        adaptive ? controller : nothing,
        IntegratorStats(),
        IntegratorOptions(
            dtmin = zero(tType),
            dtmax = tType(tf-t0),
            verbose = verbose,
            adaptive = adaptive,
            # maxiters = maxiters,
            callback = callbacks_internal,
            save_end = save_end,
            tstops = tstops_internal,
            saveat = saveat_internal,
            d_discontinuities = d_discontinuities_internal,
            tstops_cache = tstops,
            saveat_cache = saveat,
            d_discontinuities_cache = d_discontinuities,
        ),
        false,
        0,
        false,
        false,
        false,
        false,
        0,
        0,
        false,
    )
    OrdinaryDiffEqCore.initialize_callbacks!(integrator)
    DiffEqBase.initialize!(integrator, integrator.cache)

    if _tstops !== nothing
        tstops = _tstops(parameter_values(integrator), prob.tspan)
        for tstop in tstops
            add_tstop!(integrator, tstop)
        end
    end

    OrdinaryDiffEqCore.handle_dt!(integrator)

    return integrator, integrator.cache
end
