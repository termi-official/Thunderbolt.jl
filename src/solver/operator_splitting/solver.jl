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

@inline @unroll function advance_solution_to!(outer_integrator::OperatorSplittingIntegrator, subintegrators::Tuple, dof_ranges::Tuple, synchronizers::Tuple, cache::LieTrotterGodunovCache, tnext)
    # We assume that the integrators are already synced
    @unpack inner_caches = cache
    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        synchronizer = synchronizers[i]
        dof_range    = dof_ranges[i]
        # prepare_local_step!(uparent, subinteg)
        forward_sync_subintegrator!(outer_integrator, subinteg, dof_range, synchronizer)
        advance_solution_to!(outer_integrator, subinteg, dof_range, synchronizer, inner_caches[i], tnext)
        # finalize_local_step!(uparent, subinteg)
        backward_sync_internal!(outer_integrator, subinteg, dof_range)
    end
end 
