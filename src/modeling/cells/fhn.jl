
"""
The classical neuron electrophysiology model independently found by [Fit:1961:ips](@citet) and [NagARiYos:1962:apt](@citet).
This model is less stiff and cheaper than any cardiac electrophysiology model, which maks it 
a good choice for quick testing if things work at all.
"""
Base.@kwdef struct ParametrizedFHNModel{T} <: AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T = T(0.01)
end;

const FHNModel = ParametrizedFHNModel{Float64};

num_states(::ParametrizedFHNModel{T}) where{T} = 1
default_initial_state(::ParametrizedFHNModel{T}) where {T} = [0.0, 0.0]

function cell_rhs!(du::TD,φₘ::TV,s::TS,x::TX,t::TT,cell_parameters::TP) where {TD,TV,TS,TX,TT,TP <: AbstractIonicModel}
    dφₘ = @view du[1:1]
    reaction_rhs!(dφₘ,φₘ,s,x,t,cell_parameters)

    ds = @view du[2:end]
    state_rhs!(ds,φₘ,s,x,t,cell_parameters)

    return nothing
end

@inline function reaction_rhs!(dφₘ::TD,φₘ::TV,s::TS,x::TX,t::TT,cell_parameters::FHNModel) where {TD<:SubArray,TV,TS,TX,TT}
    @unpack a = cell_parameters
    dφₘ .= φₘ*(1-φₘ)*(φₘ-a) -s[1]
    return nothing
end

@inline function state_rhs!(ds::TD,φₘ::TV,s::TS,x::TX,t::TT,cell_parameters::FHNModel) where {TD<:SubArray,TV,TS,TX,TT}
    @unpack b,c,d,e = cell_parameters
    ds .= e*(b*φₘ - c*s[1] - d)
    return nothing
end
