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

@inline get_parent_index(integ::ThunderboltIntegrator, local_idx::Int) = get_parent_index(integ, local_idx, integ.indexset)
@inline get_parent_index(integ::ThunderboltIntegrator, local_idx::Int, indexset::AbstractVector) = indexset[local_idx]
@inline get_parent_index(integ::ThunderboltIntegrator, local_idx::Int, range::AbstractUnitRange) = first(range) + local_idx - 1
@inline get_parent_index(integ::ThunderboltIntegrator, local_idx::Int, range::StepRange) = first(range) + range.step*(local_idx - 1)

@inline get_parent_value(integ::ThunderboltIntegrator, local_idx::Int) = integ.uparent[get_parent_index(integ, local_idx)]

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
