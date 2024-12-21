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

struct LieTrotterGodunovCache{uType, uprevType, iiType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    inner_caches::iiType
end

function init_cache(f::GenericSplitFunction, alg::LieTrotterGodunov;
    uprev::AbstractArray, u::AbstractVector,
    inner_caches,
    alias_uprev = true,
    alias_u     = false,
)
    _uprev = alias_uprev ? uprev : SciMLBase.recursivecopy(uprev)
    _u     = alias_u     ? u     : SciMLBase.recursivecopy(u)
    LieTrotterGodunovCache(_u, _uprev, inner_caches)
end

@inline @unroll function advance_solution_to!(subintegrators::Tuple, cache::LieTrotterGodunovCache, tnext; uparent)
    # We assume that the integrators are already synced
    @unpack u, uprev, inner_caches = cache

    # Store current solution
    uprev .= u

    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        prepare_local_step!(uparent, subinteg)
        advance_solution_to!(subinteg, inner_caches[i], tnext; uparent)
        finalize_local_step!(uparent, subinteg)
    end
end 
