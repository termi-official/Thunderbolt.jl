"""
    MTKLumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct MTKLumpedCicuitModel{ProbType <: ModelingToolkit.ODEProblem} <: AbstractLumpedCirculatoryModel
    prob::ProbType
    pressure_symbols::Vector{ModelingToolkit.Num}
end

function MTKLumpedCicuitModel(sys::ModelingToolkit.ODESystem, u0, pressure_symbols::Vector{ModelingToolkit.Num})
    # To construct the ODEProblem we need to provide an initial value for the pressures
    ps = [
        sym => 0.0 for sym in pressure_symbols 
    ]
    prob = ModelingToolkit.ODEProblem(sys, u0, (0.0, 0.0), ps)
    return MTKLumpedCicuitModel(prob, pressure_symbols)
end

num_states(model::MTKLumpedCicuitModel) = length(model.prob.u0)
num_unknown_pressures(model::MTKLumpedCicuitModel) = length(model.pressure_symbols)
function get_variable_symbol_index(model::MTKLumpedCicuitModel, symbol::ModelingToolkit.Num)
    ModelingToolkit.variable_index(model.prob, symbol)
end

function default_initial_condition!(u, model::MTKLumpedCicuitModel)
    u .= model.prob.u0
end

function lumped_driver!(du, u, t, external_input::AbstractVector, model::MTKLumpedCicuitModel)
    # Update RHS
    for i in 1:length(external_input)
        model.prob.p[ModelingToolkit.parameter_index(model.prob, model.pressure_symbols[i])] = external_input[i]
    end
    # Evaluate
    model.prob.f(du, u, model.prob.p, t)
end
