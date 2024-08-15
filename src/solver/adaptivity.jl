abstract type AbstractTimeAdaptionAlgorithm end

"""
    ReactionTangentController{T <: Real} <: AbstractTimeAdaptionAlgorithm
A timestep length controller for [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite)
operator splitting using the reaction tangent as proposed in [OgiBalPer:2023:seats](@cite)
# Fields
- `σ_s::T`: steepness
- `σ_c::T`: offset in R axis
- `Δt_bounds::NTuple{2,T}`: lower and upper timestep length bounds
- `Rₙ₊₁::T`: updated maximal reaction magnitude
- `Rₙ::T`: previous reaction magnitude
"""
mutable struct ReactionTangentController{T <: Real} <: AbstractTimeAdaptionAlgorithm
    const σ_s::T
    const σ_c::T
    const Δt_bounds::NTuple{2,T}
    Rₙ₊₁::T
    Rₙ::T
end

function ReactionTangentController(σ_s::T, σ_c::T, Δt_bounds::NTuple{2,T}) where {T <: Real}
    return ReactionTangentController(σ_s, σ_c, Δt_bounds, 0.0, 0.0)
end


"""
    AdaptiveOperatorSplittingAlgorithm{TOperatorSplittingAlg <: OS.AbstractOperatorSplittingAlgorithm, TTimeAdaptionAlgorithm <: AbstractTimeAdaptionAlgorithm} <: OS.AbstractOperatorSplittingAlgorithm
A generic operator splitting algorithm `operator_splitting_algorithm` with adaptive timestepping using the controller `controller`.
# Fields
- `operator_splitting_algorithm::TOperatorSplittingAlg`: steepness
- `controller::TTimeAdaptionAlgorithm`: offset in R axis
"""
struct AdaptiveOperatorSplittingAlgorithm{TOperatorSplittingAlg <: OS.AbstractOperatorSplittingAlgorithm, TTimeAdaptionAlgorithm <: AbstractTimeAdaptionAlgorithm} <: OS.AbstractOperatorSplittingAlgorithm
    operator_splitting_algorithm::TOperatorSplittingAlg
    controller::TTimeAdaptionAlgorithm
end

@inline DiffEqBase.isadaptive(::AdaptiveOperatorSplittingAlgorithm) = true

"""
    get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
Returns the maximal reaction magnitude using the [`PointwiseODEFunction`](@ref) of an operator splitting integrator that uses [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
It is assumed that the problem containing the reaction tangent is a [`PointwiseODEFunction`](@ref).
"""
@inline function get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
    reaction_subintegrators = unrolled_filter(subintegrator -> subintegrator.f isa PointwiseODEFunction, integrator.subintegrators)
    isempty(reaction_subintegrators) && @error "PointwiseODEFunction not found"
    _get_reaction_tangent(reaction_subintegrators)
end

@inline #=@unroll=# function _get_reaction_tangent(subintegrators)
    subintegrator = subintegrators[1]
    # @unroll for subintegrator in subintegrators
        #TODO: It should be either all the the same type or just one subintegrator after filtering, don't unroll?
    φₘidx = transmembranepotential_index(subintegrator.f.ode)
    return maximum(@view subintegrator.cache.dumat[:, φₘidx])
    # end
end

@inline function OS.stepsize_controller!(integrator::OS.OperatorSplittingIntegrator, alg::AdaptiveOperatorSplittingAlgorithm)
    OS.stepsize_controller!(integrator, alg.controller, alg)
end

@inline function OS.step_accept_controller!(integrator::OS.OperatorSplittingIntegrator, alg::AdaptiveOperatorSplittingAlgorithm, q)
    OS.step_accept_controller!(integrator, alg.controller, alg, q)
end

@inline function OS.step_reject_controller!(integrator::OS.OperatorSplittingIntegrator, alg::AdaptiveOperatorSplittingAlgorithm, q)
    OS.step_reject_controller!(integrator, alg.controller, alg, q)
end

@inline function OS.stepsize_controller!(integrator::OS.OperatorSplittingIntegrator, controller::ReactionTangentController, alg::AdaptiveOperatorSplittingAlgorithm{<:OS.LieTrotterGodunov})
    @unpack σ_s, σ_c, Δt_bounds, Rₙ₊₁, Rₙ = controller
    controller.Rₙ = controller.Rₙ₊₁
    controller.Rₙ₊₁ = get_reaction_tangent(integrator)
    return nothing
end

@inline function OS.step_accept_controller!(integrator::OS.OperatorSplittingIntegrator, controller::ReactionTangentController, alg::AdaptiveOperatorSplittingAlgorithm{<:OS.LieTrotterGodunov}, q)
    @unpack σ_s, σ_c, Δt_bounds, Rₙ₊₁, Rₙ = controller
    R = max(Rₙ, Rₙ₊₁)
    integrator.dt = (1 - 1/(1+exp((σ_c - R)*σ_s)))*(Δt_bounds[2] - Δt_bounds[1]) + Δt_bounds[1]
    return nothing
end

@inline function OS.step_reject_controller!(integrator::OS.OperatorSplittingIntegrator, controller::ReactionTangentController, alg::AdaptiveOperatorSplittingAlgorithm{<:OS.LieTrotterGodunov}, q)
    return nothing # Do nothing
end

# Dispatch for outer construction
function OS.init_cache(prob::OS.OperatorSplittingProblem, alg::AdaptiveOperatorSplittingAlgorithm; dt, kwargs...) # TODO
    OS.init_cache(prob, alg.operator_splitting_algorithm;dt = dt, kwargs...)
end

# Dispatch for recursive construction
function OS.construct_inner_cache(f::OS.AbstractOperatorSplitFunction, alg::AdaptiveOperatorSplittingAlgorithm, u::AbstractArray, uprev::AbstractArray)
    OS.construct_inner_cache(f, alg.operator_splitting_algorithm, u, uprev)
end