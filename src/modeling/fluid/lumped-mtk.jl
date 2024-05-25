"""
    MTKLumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct MTKLumpedCicuitModel{SYS,F,U0} <: AbstractLumpedCirculatoryModel
    f::F
    u0::U0
    sys::SYS
    pressure_symbols::Vector{Symbol}
end

num_states(::MTKLumpedCicuitModel) = 0
num_unknown_pressures(model::MTKLumpedCicuitModel) = 0
function get_variable_symbol_index(model::MTKLumpedCicuitModel, symbol::Symbol)
    # @unpack lv_pressure_given, la_pressure_given, ra_pressure_given, rv_pressure_given = model

    # # Try to query index
    # symbol == :Vₗₐ && return lumped_circuit_relative_la_pressure_index(model)
    # symbol == :Vₗᵥ && return lumped_circuit_relative_lv_pressure_index(model)
    # symbol == :Vᵣₐ && return lumped_circuit_relative_ra_pressure_index(model)
    # symbol == :Vᵣᵥ && return lumped_circuit_relative_rv_pressure_index(model)

    # Diagnostics
    valid_symbols = Set{Symbol}()
    # la_pressure_given && push!(valid_symbols, :Vₗₐ)
    # lv_pressure_given && push!(valid_symbols, :Vₗᵥ)
    # ra_pressure_given && push!(valid_symbols, :Vᵣₐ)
    # rv_pressure_given && push!(valid_symbols, :Vᵣᵥ)
    @error "Variable named '$symbol' not found. The following symbols are defined and accessible: $valid_symbols."
end

function default_initial_condition!(u, model::MTKLumpedCicuitModel)
    # @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = model
    # u .= 0.0
    # # u[1] = V0ₗₐ
    # # u[3] = V0ᵣₐ
    # # u[4] = V0ᵣᵥ

    # # u[1] = 20.0
    # # u[2] = 500.0
    # # u[3] = 20.0
    # # u[4] = 500.0

    # # TODO obtain via pre-pacing in isolation
    # u .= [31.78263930696728, 20.619293500582675, 76.99868985499684, 28.062792020353495, 5.3259599006276925, 13.308990108813674, 1.848880514855276, 3.6948263599349302, -9.974721253140004, -17.12404226311947, -11.360818019572653, -19.32908606755043]
end


function lumped_driver!(du, u, t, external_input::AbstractVector, model::MTKLumpedCicuitModel)
    # Translate the external
    p = [
        query_mtk_parameter_by_symbol(model.sys, model.pressure_symbols[i]) => external_input[i] for i in 1:length(external_input)
    ]

    return model.f(du, u, p, t)
end
