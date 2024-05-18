# This file contains the main infrastucture for coupled problems

struct EmptyTyingCache end

get_tying_dofs(::EmptyTyingCache, u) = nothing

assemble_tying!(Jₑ, residualₑ, uₑ, uₜ, cell,::EmptyTyingCache, t) = nothing
assemble_tying!(Jₑ, uₑ, uₜ, cell,::EmptyTyingCache, t) = nothing

function setup_tying_cache(tying_models::Union{<:AbstractVector,<:Tuple}, qr, ip, ip_geo)
    length(tying_models) == 0 && return EmptyTyingCache()
    return ntuple(i->setup_tying_cache(tying_models[i], qr, ip, ip_geo), length(tying_models))
end
