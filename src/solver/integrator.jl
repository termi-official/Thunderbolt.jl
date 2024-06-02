#TODO rename into something with Time due to name collision with other integrators
"""
Internal helper to integrate a single inner operator
over some time interval.
"""
mutable struct ThunderboltIntegrator{
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
}  <: DiffEqBase.SciMLBase.DEIntegrator{#=alg_type=#Nothing, true, uType, tType}
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
end

# TODO remove me
const ThunderboltSubIntegrator = ThunderboltIntegrator

# TimeChoiceIterator API
@inline function DiffEqBase.get_tmp_cache(integrator::ThunderboltIntegrator)
    return (integrator.cache.tmp,)
end
@inline function DiffEqBase.get_tmp_cache(integrator::ThunderboltIntegrator, alg, cache)
    return (cache.tmp,)
end

# Interpolation
# TODO via https://github.com/SciML/SciMLBase.jl/blob/master/src/interpolation.jl
# TODO deduplicate with OS module
function (integrator::ThunderboltIntegrator)(tmp, t)
    OS.linear_interpolation!(tmp, t, integrator.uprev, integrator.u, integrator.t-integrator.dt, integrator.t)
end
