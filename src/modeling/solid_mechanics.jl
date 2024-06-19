
"""
    StructuralModel(mechanical_model, face_models)

A generic model for structural problems.
"""
struct StructuralModel{MM, FM}
    displacement_symbol::Symbol
    mechanical_model::MM
    face_models::FM
end

get_field_variable_names(model::StructuralModel) = (model.displacement_symbol, )

include("solid/energies.jl")
include("solid/contraction.jl")
include("solid/active.jl")
include("solid/materials.jl")
include("solid/elements.jl")
