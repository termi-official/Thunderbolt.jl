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

@inline @unroll function advance_solution_to!(outer_integrator::OperatorSplittingIntegrator, subintegrators::Tuple, solution_indices::Tuple, synchronizers::Tuple, cache::LieTrotterGodunovCache, tnext)
    # We assume that the integrators are already synced
    @unpack inner_caches = cache
    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        synchronizer = synchronizers[i]
        idxs         = solution_indices[i]
        cache        = inner_caches[i]

        # prepare_local_step!(uparent, subinteg)
        @timeit_debug "suboperator $i" begin
            @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
            @timeit_debug "time solve" advance_solution_to!(outer_integrator, subinteg, idxs, synchronizer, cache, tnext)
            if !(subinteg isa Tuple) && subinteg.sol.retcode ∉ (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
                return
            end
            # finalize_local_step!(uparent, subinteg)
            @timeit_debug "sync <-" backward_sync_subintegrator!(outer_integrator, subinteg, idxs)
        end
    end
end 
