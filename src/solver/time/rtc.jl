"""
    ReactionTangentController{LTG <: OS.LieTrotterGodunov, T <: Real} <: OS.AbstractOperatorSplittingAlgorithm
A timestep length controller for [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite)
operator splitting using the reaction tangent as proposed in [OgiBalPer:2023:seats](@cite)
The next timestep length is calculated as
```math
\\sigma\\left(R_{\\max }\\right):=\\left(1.0-\\frac{1}{1+\\exp \\left(\\left(\\sigma_{\\mathrm{c}}-R_{\\max }\\right) \\cdot \\sigma_{\\mathrm{s}}\\right)}\\right) \\cdot\\left(\\Delta t_{\\max }-\\Delta t_{\\min }\\right)+\\Delta t_{\\min }
```
# Fields
- `ltg`::`LTG`: `LieTrotterGodunov` algorithm
- `σ_s::T`: steepness
- `σ_c::T`: offset in R axis
- `Δt_bounds::NTuple{2,T}`: lower and upper timestep length bounds
"""
struct ReactionTangentController{LTG <: OS.LieTrotterGodunov, T <: Real} <: OS.AbstractOperatorSplittingAlgorithm
    ltg::LTG
    σ_s::T
    σ_c::T
    Δt_bounds::NTuple{2,T}
end

mutable struct ReactionTangentControllerCache{T <: Real, LTGCache <: OS.LieTrotterGodunovCache, uType} <: OS.AbstractOperatorSplittingCache
    const ltg_cache::LTGCache
    u::uType
    uprev::uType # True previous solution
    Rₙ₊₁::T
    Rₙ::T
    function ReactionTangentControllerCache(ltg_cache::LTGCache, Rₙ₊₁::T, Rₙ::T) where {T, LTGCache <: OS.LieTrotterGodunovCache}
        uType = typeof(ltg_cache.u)
        return new{T, LTGCache, uType}(ltg_cache, ltg_cache.u, ltg_cache.uprev, Rₙ₊₁, Rₙ)
    end
end

@inline DiffEqBase.get_tmp_cache(integrator::OS.OperatorSplittingIntegrator, alg::OS.AbstractOperatorSplittingAlgorithm, cache::ReactionTangentControllerCache) = DiffEqBase.get_tmp_cache(integrator, alg, cache.ltg_cache)

@inline function OS.advance_solution_to!(subintegrators::Tuple, cache::ReactionTangentControllerCache, tnext) 
    OS.advance_solution_to!(subintegrators, cache.ltg_cache, tnext)
end

@inline DiffEqBase.isadaptive(::ReactionTangentController) = true

"""
    get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
Returns the maximal reaction magnitude using the [`PointwiseODEFunction`](@ref) of an operator splitting integrator that uses [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
It is assumed that the problem containing the reaction tangent is a [`PointwiseODEFunction`](@ref).
"""
@inline function get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
    R, _ = _get_reaction_tangent(integrator.subintegrators)
    return R
end

@inline @unroll function _get_reaction_tangent(subintegrators, n_reaction_tangents::Int = 0)
    R = 0.0
    @unroll for subintegrator in subintegrators
        if subintegrator isa Tuple
            R, n_reaction_tangents = _get_reaction_tangent(subintegrator, n_reaction_tangents)
        elseif subintegrator.f isa PointwiseODEFunction
            n_reaction_tangents += 1
            φₘidx = transmembranepotential_index(subintegrator.f.ode)
            R = max(R, maximum(@view subintegrator.cache.dumat[:, φₘidx]))
        end
    end
    @assert n_reaction_tangents == 1 "No or multiple integrators using PointwiseODEFunction found"
    return (R, n_reaction_tangents)
end

@inline function OS.stepsize_controller!(integrator::OS.OperatorSplittingIntegrator, alg::ReactionTangentController)
    integrator.cache.Rₙ = integrator.cache.Rₙ₊₁
    integrator.cache.Rₙ₊₁ = get_reaction_tangent(integrator)
    return nothing
end

@inline function OS.step_accept_controller!(integrator::OS.OperatorSplittingIntegrator, alg::ReactionTangentController, q)
    @unpack Rₙ₊₁, Rₙ = integrator.cache
    @unpack σ_s, σ_c, Δt_bounds = alg
    R = max(Rₙ, Rₙ₊₁)
    if isinf(σ_s) && R == σ_c
        # Handle it?
        throw(ErrorException("Δt is undefined for R = σ_c where σ_s = ∞"))
    end
    integrator._dt = (1 - 1/(1+exp((σ_c - R)*σ_s)))*(Δt_bounds[2] - Δt_bounds[1]) + Δt_bounds[1]
    return nothing
end

@inline function OS.step_reject_controller!(integrator::OS.OperatorSplittingIntegrator, alg::ReactionTangentController, q)
    return nothing # Do nothing
end

# Dispatch for outer construction
function OS.init_cache(prob::OS.OperatorSplittingProblem, alg::ReactionTangentController; dt, kwargs...)
    @unpack f = prob
    @assert f isa GenericSplitFunction

    u          = copy(prob.u0)
    uprev      = copy(prob.u0)

    # Build inner integrator
    return OS.construct_inner_cache(f, alg, u, uprev)
end

# Dispatch for recursive construction
function OS.construct_inner_cache(f::OS.AbstractOperatorSplitFunction, alg::ReactionTangentController, u::AbstractArray{T}, uprev::AbstractArray) where T <: Number
    ltg_cache = OS.construct_inner_cache(f, alg.ltg, u, uprev)
    return ReactionTangentControllerCache(ltg_cache, zero(T), zero(T))
end

function OS.build_subintegrators_recursive(f::GenericSplitFunction, synchronizers::Tuple, p::Tuple, cache::ReactionTangentControllerCache, u::AbstractArray, uprev::AbstractArray, t, dt, dof_range, uparent, tstops, _tstops, saveat, _saveat)
    OS.build_subintegrators_recursive(f, synchronizers, p, cache.ltg_cache, u, uprev, t, dt, dof_range, uparent, tstops, _tstops, saveat, _saveat)
end
