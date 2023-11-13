abstract type InterfaceCoupler end

abstract type VolumeCoupler end

struct Coupling{CouplerType}
    problem_1_index::Int
    problem_2_index::Int
    coupler::CouplerType
end

struct CoupledModel{MT, CT}
    base_models::MT
    couplers::CT
end
