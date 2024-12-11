mutable struct IntegratorStats
    naccept::Int64
    nreject::Int64
    # TODO inner solver stats
end

IntegratorStats() = IntegratorStats(0,0)

Base.@kwdef  struct IntegratorOptions{tType, #=msgType,=# F1, F2, F3, F4, F5, SType, tstopsType, discType, tcache, savecache, disccache}
    dtmin::tType = eps(tType)
    dtmax::tType = Inf
    failfactor::tType = 4.0
    verbose::Bool = false
    adaptive::Bool = false # Redundant with the dispatch on DiffEqBase.isadaptive below (alg adaptive + controller not nothing)
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
    saveat::tstopsType = nothing
    save_end::Bool = true
    d_discontinuities::discType = nothing
    tstops_cache::tcache = nothing
    saveat_cache::savecache = nothing
    d_discontinuities_cache::disccache = nothing
    callback::F5
    dense::Bool = false
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
    callback_cache::callbackcacheType
    synchronizer::syncType
    sol::solType
    const dtchangeable::Bool
    tstops_internal::heapType
    tstops::tstopsType # argument to __init used as default argument to reinit!
    saveat::heapType
    saveat_internal::saveatType # argument to __init used as default argument to reinit!
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
end

DiffEqBase.has_stats(::ThunderboltTimeIntegrator) = true

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

function DiffEqBase.last_step_failed(integrator::ThunderboltTimeIntegrator)
    integrator.last_step_failed && !integrator.opts.adaptive
end

# Solution interface
function should_accept_step(integrator::ThunderboltTimeIntegrator)
    if integrator.force_stepfail || integrator.isout
        return false
    end
    return should_accept_step(integrator, integrator.cache, integrator.controller)
end
function should_accept_step(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    return !(integrator.force_stepfail)
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
    if integrator.iter > 0
        if should_accept_step(integrator)
            accept_step!(integrator)
        else # Step should be rejected and hence repeated
            reject_step!(integrator)
        end
    elseif integrator.u_modified # && integrator.iter == 0
        update_uprev!(integrator)
    end

    # Before stepping we might need to adjust the dt
    integrator.iter += 1
    OrdinaryDiffEqCore.choose_algorithm!(integrator, integrator.cache)
    OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator)
    OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator)
    integrator.force_stepfail = false
end

function update_uprev!(integrator::ThunderboltTimeIntegrator)
    integrator.uprev .= integrator.u
end

function footer_reset_flags!(integrator)
    integrator.u_modified = false
end

OrdinaryDiffEqCore.postamble!(integrator::ThunderboltTimeIntegrator) = OrdinaryDiffEqCore._postamble!(integrator)

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
        # OrdinaryDiffEqCore.handle_callbacks!(integrator)
        adapt_dt!(integrator) # Noop for non-adaptive algorithms
    elseif integrator.force_stepfail
        if DiffEqBase.isadaptive(integrator)
            OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
        # elseif integrator.dtchangeable # Non-adaptive but can change dt
        #     integrator.dt *= integrator.opts.failfactor
        elseif integrator.last_step_failed
            return
        end
        integrator.last_step_failed = true
    end

    # if integrator.opts.progress && integrator.iter % integrator.opts.progress_steps == 0
    #     log_step!(integrator.opts.progress_name, integrator.opts.progress_id,
    #         integrator.opts.progress_message, integrator.dt, integrator.u,
    #         integrator.p, integrator.t, integrator.sol.prob.tspan)
    # end
    nothing
end

# function handle_tstop!(integrator::ThunderboltTimeIntegrator)
#     (; tstops) = integrator
#     while !isempty(tstops) && OS.reached_tstop(integrator, first(tstops))
#         pop!(tstops)
#     end
# end

# function fix_dt_at_bounds!(integrator::ThunderboltTimeIntegrator)
#     if integrator.tdir > 0
#         integrator.dt = min(integrator.opts.dtmax, integrator.dt)
#     else
#         integrator.dt = max(integrator.opts.dtmax, integrator.dt)
#     end
#     dtmin = DiffEqBase.timedepentdtmin(integrator.t, integrator.opts.dtmin)
#     if integrator.tdir > 0
#         integrator.dt = max(integrator.dt, dtmin)
#     else
#         integrator.dt = min(integrator.dt, dtmin)
#     end
#     return nothing
# end

# function modify_dt_for_tstops!(integrator::ThunderboltTimeIntegrator)
#     if DiffEqBase.has_tstop(integrator) && integrator.dtchangeable
#         tdir_t = integrator.tdir * integrator.t
#         tdir_tstop = DiffEqBase.first_tstop(integrator)
#         dttstop = abs(tdir_tstop - tdir_t)
#         # if dttstop < DiffEqBase.timedepentdtmin(integrator.t, integrator.opts.dtmin)
#         #     integrator.t = tdir_tstop
#         # else
#             integrator.dt = integrator.tdir *
#                             min(abs(integrator.dt), abs(dttstop))
#         # end
#     end
# end

# function DiffEqBase.step!(integrator::ThunderboltTimeIntegrator, dt, stop_at_tdt = false)
#     dt <= zero(dt) && error("dt must be positive")
#     tstop = integrator.t + dt
#     while !OS.reached_tstop(integrator, tstop, stop_at_tdt)
#         @show integrator.t, tstop
#         step_header!(integrator)
#         if !perform_step!(integrator, integrator.cache)
#             integrator.force_stepfail = true
#         end
#         step_footer!(integrator)
#     end

#     handle_tstop!(integrator)
# end

@inline function OrdinaryDiffEqCore.step!(integrator::ThunderboltTimeIntegrator)
    @inbounds step_header!(integrator)
    if OrdinaryDiffEqCore.check_error!(integrator) != DiffEqBase.ReturnCode.Success
        return
    end
    @inbounds OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
    @inbounds step_footer!(integrator)
    @inbounds while !should_accept_step(integrator)
        step_header!(integrator)
        if OrdinaryDiffEqCore.check_error!(integrator) != DiffEqBase.ReturnCode.Success
            return
        end
        OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
        step_footer!(integrator)
    end
    @inbounds OrdinaryDiffEqCore.handle_tstop!(integrator)
end

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
    # dt ≈ 0.0 || DiffEqBase.step!(integrator, dt, true)
    @inbounds while integrator.tdir * integrator.t < tend
        step_header!(integrator)
        if OrdinaryDiffEqCore.check_error!(integrator) != DiffEqBase.ReturnCode.Success
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

struct DummyODESolution <: DiffEqBase.AbstractODESolution{Float64, 2, Vector{Float64}}
    retcode::DiffEqBase.ReturnCode.T
end
DummyODESolution() = DummyODESolution(DiffEqBase.ReturnCode.Default)
function DiffEqBase.solution_new_retcode(sol::DummyODESolution, retcode)
    return DiffEqBase.@set sol.retcode = retcode
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
        nothing,
        synchronizer,
        DummyODESolution(),
        true, #dtchangeable
        tstops,
        _tstops,
        saveat,
        _saveat,
        nothing, # FIXME controller
        IntegratorStats(),
        IntegratorOptions(; # TODO pass from outside
            adaptive = false,
            dtmin = eps(tType),
            dtmax = tType(Inf),
            callback = DiffEqBase.CallbackSet(nothing),
        ),
        false,
        0,
        false,
        false,
        false,
        false,
        0,
        0,
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
    alias_du0 = false,
    controller = nothing,
    maxiters = 1000000,
    dense = save_everystep &&
                    !(alg isa DAEAlgorithm) && !(prob isa DiscreteProblem) &&
                    isempty(saveat),
    save_func = (u, t) -> copy(u),                  # custom kwarg
    dtchangeable = true,                            # custom kwarg
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

    cache = init_cache(prob, alg)

    # Setup solution buffers
    u  = setup_u(prob, alg, alias_u0)
    uType                = typeof(u)
    uBottomEltype        = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)

    rate_prototype = compute_rate_prototype(prob, alias_du0)
    rateType = typeof(rate_prototype)

    # Setup callbacks
    callbacks_internal = DiffEqBase.CallbackSet(callback)
    max_len_cb = DiffEqBase.max_vector_callback_length_int(callbacks_internal)
    if max_len_cb !== nothing
        uBottomEltypeReal = real(uBottomEltype)
        if DiffEqBase.isinplace(prob)
            callback_cache = DiffEqBase.CallbackCache(u, max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        else
            callback_cache = DiffEqBase.CallbackCache(max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        end
    else
        callback_cache = nothing
    end

    # Setup solution
    save_idxs, saved_subsystem = DiffEqBase.SciMLBase.get_save_idxs_and_saved_subsystem(prob, save_idxs)

    if save_idxs === nothing
        ksEltype = Vector{rateType}
    else
        ks_prototype = rate_prototype[save_idxs]
        ksEltype = Vector{typeof(ks_prototype)}
    end

    ts = ts_init === () ? tType[] : convert(Vector{tType}, ts_init)
    ks = ks_init === () ? ksEltype[] : convert(Vector{ksEltype}, ks_init)

    sol = DiffEqBase.build_solution(
        prob, alg, ts, uType[],
        dense = dense, k = ks, saved_subsystem = saved_subsystem,
        calculate_error = false
    )

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
        tstops_internal,
        tstops,
        saveat_internal,
        saveat,
        adaptive ? controller : nothing,
        IntegratorStats(),
        IntegratorOptions(
            dtmin = eps(tType),
            dtmax = tType(tf-t0),
            verbose = verbose,
            adaptive = adaptive,
            maxiters = maxiters,
            callback = callbacks_internal,
            save_end = save_end,
            tstops = tstops,
            saveat = saveat,
            d_discontinuities = d_discontinuities_internal,
            tstops_cache = tstops_internal,
            saveat_cache = saveat_internal,
        ),
        false,
        0,
        false,
        false,
        false,
        false,
        0,
        0,
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

    return integrator
end

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
        return recursivecopy(u)
    else
        _compute_rate_prototype_mass_matrix_form(prob)
    end
end
function compute_rate_prototype(prob::DiffEqBase.DiscreteProblem)
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
function compute_rate_prototype(prob::DiffEqBase.DAEProblem)
    return prob.du0
end

function compute_rate_prototype(prob::AbstractSemidiscreteProblem)
    _compute_rate_prototype_mass_matrix_form(prob)
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
    integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, DiffEqBase.DiffEqBase.ReturnCode.Success)
    return integrator.sol
end

# Some common getters
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int) = get_parent_index(integ, local_idx, integ.indexset)
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int, indexset::AbstractVector) = indexset[local_idx]
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int, range::AbstractUnitRange) = first(range) + local_idx - 1
@inline get_parent_index(integ::ThunderboltTimeIntegrator, local_idx::Int, range::StepRange) = first(range) + range.step*(local_idx - 1)

# Compat with OrdinaryDiffEq
function OrdinaryDiffEqCore.perform_step!(integ::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache)
    integ.opts.verbose && @info "Time integration on [$(integ.t), $(integ.t+integ.dt)] (Δt=$(integ.dt))"
    if !perform_step!(integ.f, cache, integ.t, integ.dt)
        # if integ.sol !== nothing # FIXME
        #     integ.sol = DiffEqBase.solution_new_retcode(integ.sol, DiffEqBase.DiffEqBase.ReturnCode.Failure)
        # end
        # return false
        integ.force_stepfail = true
    end
    # return true
end

function init_cache(prob, alg; dt, kwargs...)
    return setup_solver_cache(prob.f, alg, prob.tspan[1])
end

function setup_u(prob::AbstractSemidiscreteProblem, solver, alias_u0)
    if alias_u0
        return prob.u0
    else
        return recursivecopy(prob.u0)
    end
end

OrdinaryDiffEqCore.choose_algorithm!(integrator, cache::AbstractTimeSolverCache) = nothing
