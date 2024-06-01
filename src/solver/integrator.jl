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
    solType,
}  <: DiffEqBase.SciMLBase.DEIntegrator{#=alg_type=#Nothing, true, uType, tType}
    f::fType # Right hand side
    u::uType # Current local solution
    umaster::uType2 # Real solution injected by OperatorSplittingIntegrator
    uprev::uprevType
    indexset::indexSetType
    p::pType
    t::tType
    tprev::tType
    dt::tType
    cache::cacheType
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

@inline function step_begin!(subintegrator::ThunderboltIntegrator)
    # Copy solution into subproblem
    for (i,imain) in enumerate(subintegrator.indexset)
        subintegrator.u[i] = subintegrator.umaster[imain]
    end
    # Mark previous solution
    subintegrator.uprev .= subintegrator.u
end

@inline function step_end!(subintegrator::ThunderboltIntegrator)
    # Copy solution out of subproblem
    for (i,imain) in enumerate(subintegrator.indexset)
        subintegrator.umaster[imain] = subintegrator.u[i]
    end
end

# Interpolation
# TODO via https://github.com/SciML/SciMLBase.jl/blob/master/src/interpolation.jl
# TODO deduplicate with OS module
function linear_interpolation!(y,t,y1,y2,t1,t2)
    y .= y1 + (t-t1) * (y2-y1)/(t2-t1)
end
function (integrator::ThunderboltIntegrator)(tmp, t)
    linear_interpolation!(tmp, t, integrator.uprev, integrator.u, integrator.t-integrator.dt, integrator.t)
end