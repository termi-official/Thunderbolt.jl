abstract type AbstractCoupler end

"""
Abstract supertype for all interface coupling schemes.
"""
abstract type InterfaceCoupler <: AbstractCoupler end

"""
Abstract supertype for all volume coupling schemes.
"""
abstract type VolumeCoupler <: AbstractCoupler end

"""
Helper to describe the coupling between problems.
"""
struct Coupling{CouplerType <: AbstractCoupler}
    problem_1_index::Int
    problem_2_index::Int
    coupler::CouplerType
end

"""
A descriptor for a coupled model.
"""
struct CoupledModel{MT <: Tuple, CT <: Tuple}
    base_models::MT
    couplers::CT
end

CoupledModel(base_models::Tuple, coupler::Coupling) = CoupledModel(base_models, (coupler,))
CoupledModel(base_models::AbstractVector, coupler::Coupling) = CoupledModel(ntuple(i -> base_models[i], length(base_models)), (coupler,))
CoupledModel(base_models::AbstractVector, couplers::Tuple) = CoupledModel(ntuple(i -> base_models[i], length(base_models)), couplers)
CoupledModel(base_models::Tuple, couplers::AbstractVector) = CoupledModel(base_models, ntuple(i -> couplers[i], length(couplers)))
CoupledModel(base_models::AbstractVector, couplers::AbstractVector) = CoupledModel(ntuple(i -> base_models[i], length(base_models)), ntuple(i -> couplers[i], length(couplers)))
