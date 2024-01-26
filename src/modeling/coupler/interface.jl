"""
Abstract supertype for all interface coupling schemes.
"""
abstract type InterfaceCoupler end

"""
Abstract supertype for all volume coupling schemes.
"""
abstract type VolumeCoupler end

"""
Helper to describe the coupling between problems.
"""
struct Coupling{CouplerType}
    problem_1_index::Int
    problem_2_index::Int
    coupler::CouplerType
end

"""
A descriptor for a coupled model.
"""
struct CoupledModel{MT, CT}
    base_models::MT
    couplers::CT
end
