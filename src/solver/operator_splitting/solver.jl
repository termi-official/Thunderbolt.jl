# Lie-Trotter-Godunov Splitting Implementation
"""
    LieTrotterGodunov <: AbstractOperatorSplittingAlgorithm
A first order operator splitting algorithm attributed to [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
"""
struct LieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
    # transfer_algs::TransferTupleType # Tuple of transfer algorithms from the master solution into the individual ones
end

@inline DiffEqBase.isadaptive(::AbstractOperatorSplittingAlgorithm) = false

struct LieTrotterGodunovCache{uType, tmpType, iiType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uType # True previous solution
    tmp::tmpType # Scratch
    inner_caches::iiType
end

# Dispatch for outer construction
function init_cache(prob::OperatorSplittingProblem, alg::LieTrotterGodunov; u0, kwargs...) # TODO
    @unpack f = prob
    @assert f isa GenericSplitFunction

    # Build inner integrator
    return construct_inner_cache(f, alg; u0, kwargs...)
end

# Dispatch for recursive construction
function construct_inner_cache(f::AbstractOperatorSplitFunction, alg::LieTrotterGodunov; u0, kwargs...)
    dof_ranges = f.dof_ranges

    u          = copy(u0)
    uprev      = copy(u0)
    tmp        = similar(u)
    inner_caches = ntuple(i->construct_inner_cache(get_operator(f, i), alg.inner_algs[i]; u0, kwargs...), length(f.functions))
    LieTrotterGodunovCache(u, uprev, tmp, inner_caches)
end

@inline @unroll function advance_solution_to!(subintegrators::Tuple, cache::LieTrotterGodunovCache, tnext)
    # We assume that the integrators are already synced
    @unpack u, uprev, inner_caches = cache

    # Store current solution
    uprev .= u

    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        prepare_local_step!(subinteg)
        advance_solution_to!(subinteg, inner_caches[i], tnext)
        finalize_local_step!(subinteg)
    end
end 
