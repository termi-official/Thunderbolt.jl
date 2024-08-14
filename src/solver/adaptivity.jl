abstract type AbstractTimeAdaptionAlgorithm end

struct NoTimeAdaption <: AbstractTimeAdaptionAlgorithm end

mutable struct ReactionTangentController{T <: Real} <: AbstractTimeAdaptionAlgorithm
    const σ_s::T
    const σ_c::T
    const Δt_bounds::NTuple{2,T}
    Rₙ₊₁::T
    Rₙ::T
end

function get_next_dt(controller::ReactionTangentController)
    @unpack σ_s, σ_c, Δt_bounds, Rₙ₊₁, Rₙ = controller
    R = max(Rₙ, Rₙ₊₁)
    (1 - 1/(1+exp((σ_c - R)*σ_s)))*(Δt_bounds[2] - Δt_bounds[1]) + Δt_bounds[1]
end

struct AdaptiveOperatorSplittingAlgorithm{TOperatorSplittingAlg <: OS.AbstractOperatorSplittingAlgorithm, TTimeAdaptionAlgorithm <: AbstractTimeAdaptionAlgorithm} <: OS.AbstractOperatorSplittingAlgorithm
    operator_splitting_algorithm::TOperatorSplittingAlg
    timestep_controller::TTimeAdaptionAlgorithm
end

is_adaptive(::AdaptiveOperatorSplittingAlgorithm) = true


@inline function get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
    for subintegrator in integrator.subintegrators
        subintegrator.f isa PointwiseODEFunction || continue
        φₘidx = transmembranepotential_index(subintegrator.f.ode)
        return maximum(@view subintegrator.cache.dumat[:, φₘidx])
    end
end


@inline function OS.update_controller!(integrator::OS.OperatorSplittingIntegrator{<:Any, <:AdaptiveOperatorSplittingAlgorithm{<:OS.LieTrotterGodunov, <:ReactionTangentController}})
    controller = integrator.alg.timestep_controller
    controller.Rₙ = controller.Rₙ₊₁
    controller.Rₙ₊₁ = get_reaction_tangent(integrator)
end

@inline function OS.update_dt!(integrator::OS.OperatorSplittingIntegrator{<:Any, <:AdaptiveOperatorSplittingAlgorithm{<:OS.LieTrotterGodunov, <:ReactionTangentController}})
    controller = integrator.alg.timestep_controller
    integrator._dt = get_next_dt(controller)
end

# Dispatch for outer construction
function OS.init_cache(prob::OS.OperatorSplittingProblem, alg::AdaptiveOperatorSplittingAlgorithm; dt, kwargs...) # TODO
    OS.init_cache(prob, alg.operator_splitting_algorithm;dt = dt, kwargs...)
end

# Dispatch for recursive construction
function OS.construct_inner_cache(f::OS.AbstractOperatorSplitFunction, alg::AdaptiveOperatorSplittingAlgorithm, u::AbstractArray, uprev::AbstractArray)
    OS.construct_inner_cache(f, alg.operator_splitting_algorithm, u, uprev)
end