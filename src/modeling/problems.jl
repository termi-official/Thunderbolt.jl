# TODO rethink interface here
#      1. Who creates the solution vector?
#      2. Is there a better way to pass the initial solution information?
default_initializer(problem, t) = error("No default initializer available for a problem of type $(typeof(problem))")

struct NullProblem
    ndofs::Int
end

default_initializer(problem::NullProblem, t) = zeros(problem.ndofs)

struct CoupledProblem{MT, CT}
    base_problems::MT
    couplers::CT
end

function default_initializer(problem::CoupledProblem, t)
    mortar([default_initializer(p,t) for p ∈ problem.base_problems])
end

# TODO support arbitrary splits
struct SplitProblem{APT, BPT}
    A::APT
    B::BPT
end

default_initializer(problem::SplitProblem, t) = (default_initializer(problem.A, t), default_initializer(problem.B, t))

# TODO support arbitrary partitioning
struct PartitionedProblem{APT, BPT}
    A::APT
    B::BPT
end

abstract type AbstractPointwiseProblem end

struct ODEProblem{ODET,F,P}
    ode::ODET
    f::F
    p::P
end

function default_initializer(problem::ODEProblem, t) 
    u = zeros(num_states(problem.ode))
    initial_condition!(u, problem.ode)
    u
end

struct PointwiseODEProblem{ODET} <: AbstractPointwiseProblem
    npoints::Int
    ode::ODET
end

default_initializer(problem::PointwiseODEProblem, t) = default_initializer(problem.ode, t)

struct TransientHeatProblem{DTF, ST, DH}
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
end

"""
    QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A discrete problem with time dependent terms and no time derivatives w.r.t. any solution variable.
Abstractly written we want to solve the problem F(u, t) = 0 on some time interval [t₁, t₂].
"""
struct QuasiStaticNonlinearProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH}
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
end

default_initializer(problem::QuasiStaticNonlinearProblem, t) = zeros(ndofs(problem.dh))

"""
    QuasiStaticODEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.

TODO implement.
"""
struct QuasiStaticODEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH}
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
end

"""
    QuasiStaticDAEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A problem with time dependent terms and time derivatives only w.r.t. internal solution variable which can't be expressed as an ODE.

TODO implement.
"""
struct QuasiStaticDAEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH}
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
end
