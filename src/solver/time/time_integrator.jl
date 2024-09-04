"""
Internal helper to integrate a single inner operator
over some time interval.
"""
mutable struct ThunderboltTimeIntegrator{
    fType,
    uType,
    uType2,
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
}  <: DiffEqBase.SciMLBase.DEIntegrator{#=alg_type=#Nothing, true, uType, tType} # FIXME alg
    f::fType # Right hand side
    u::uType # Current local solution
    uparent::uType2 # Real solution injected by OperatorSplittingIntegrator
    uprev::uprevType
    indexset::indexSetType
    p::pType
    t::tType
    tprev::tType
    dt::tType
    cache::cacheType
    synchronizer::syncType
    sol::solType
    dtchangeable::Bool
    tstops::heapType
    _tstops::tstopsType # argument to __init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType # argument to __init used as default argument to reinit!
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

function DiffEqBase.step!(integrator::ThunderboltTimeIntegrator, dt, stop_at_tdt = false)
    (; tstops) = integrator
    dt <= zero(dt) && error("dt must be positive")
    tnext = integrator.t + dt
    while !OS.reached_tstop(integrator, tnext, stop_at_tdt)
        # Solve inner problem
        perform_step!(integrator, integrator.cache) || error("Time integration failed at t=$(integrator.t).") # remove this
        # Update integrator
        integrator.tprev = integrator.t
        integrator.t = integrator.t + integrator.dt
    end

    while !isempty(tstops) && OS.reached_tstop(integrator, first(tstops))
        pop!(tstops)
    end
end
function OS.synchronize_subintegrator!(subintegrator::ThunderboltTimeIntegrator, integrator::OS.OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    subintegrator.t = t
    subintegrator.dt = dt
end
# TODO some operator splitting methods require to go back in time, so we need to figure out what the best way is.
OS.tdir(::ThunderboltTimeIntegrator) = 1

# TODO Any -> cache supertype
function OS.advance_solution_to!(integrator::ThunderboltTimeIntegrator, cache::Any, tend)
    @unpack f, t = integrator
    dt = tend-t
    dt ≈ 0.0 || DiffEqBase.step!(integrator, dt, true)
end
@inline function OS.prepare_local_step!(subintegrator::ThunderboltTimeIntegrator)
    # Copy solution into subproblem
    uparentview      = @view subintegrator.uparent[subintegrator.indexset]
    subintegrator.u .= uparentview
    # for (i,imain) in enumerate(subintegrator.indexset)
    #     subintegrator.u[i] = subintegrator.uparent[imain]
    # end
    # Mark previous solution
    subintegrator.uprev .= subintegrator.u
    syncronize_parameters!(subintegrator, subintegrator.f, subintegrator.synchronizer)
end
@inline function OS.finalize_local_step!(subintegrator::ThunderboltTimeIntegrator)
    # Copy solution out of subproblem
    #
    uparentview = @view subintegrator.uparent[subintegrator.indexset]
    uparentview .= subintegrator.u
    # for (i,imain) in enumerate(subintegrator.indexset)
    #     subintegrator.uparent[imain] = subintegrator.u[i]
    # end
end
# Glue code
function OS.build_subintegrators_recursive(f, synchronizer, p::Any, cache::AbstractTimeSolverCache, u::AbstractArray, uprev::AbstractArray, t, dt, dof_range, uparent, tstops, _tstops, saveat, _saveat)
    integrator = Thunderbolt.ThunderboltTimeIntegrator(
        f,
        cache.uₙ,
        uparent,
        cache.uₙ₋₁,
        dof_range,
        p,
        t,
        t,
        dt,
        cache,
        synchronizer,
        nothing, # FIXME sol
        true, #dtchangeable
        tstops,
        _tstops,
        saveat,
        _saveat,
    )
    # This makes sure that the parameters are set correctly for the first time step
    syncronize_parameters!(integrator, f, synchronizer)
    return integrator
end
function OS.construct_inner_cache(f, alg::AbstractSolver, u::AbstractArray, uprev::AbstractArray)
    return Thunderbolt.setup_solver_cache(f, alg, 0.0)
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
    save_func = (u, t) -> copy(u),                  # custom kwarg
    dtchangeable = true,                            # custom kwarg
    uparent = nothing,                              # custom kwarg
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

    cache = setup_solver_cache(f, alg, t0)

    cache.uₙ .= u0
    cache.uₙ₋₁ .= u0

    integrator = ThunderboltTimeIntegrator(
        f,
        cache.uₙ,
        uparent,
        cache.uₙ₋₁,
        1:length(u0),
        p,
        t0,
        t0,
        dt,
        cache,
        syncronizer,
        sol,
        true,
        tstops,
        _tstops,
        saveat,
        _saveat,
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

@inline get_parent_value(integ::ThunderboltTimeIntegrator, local_idx::Int) = integ.uparent[get_parent_index(integ, local_idx)]

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
