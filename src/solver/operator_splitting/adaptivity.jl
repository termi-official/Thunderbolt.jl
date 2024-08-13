abstract type AbstractTimeAdaptionAlgorithm end

struct NoTimeAdaption <: AbstractTimeAdaptionAlgorithm end

struct ReactionTangentController{T <: Real} <: AbstractTimeAdaptionAlgorithm
    σ_s::T
    σ_c::T
    Δt_bounds::NTuple{2,T}
end

function get_next_dt(R, controller::ReactionTangentController )
    @unpack σ_s, σ_c, Δt_bounds = controller
    (1 - 1/(1+exp((σ_c - R)*σ_s)))*(Δt_bounds[2] - Δt_bounds[1]) + Δt_bounds[1]
end
