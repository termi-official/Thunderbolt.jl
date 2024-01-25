# TODO rethink interface here
#      1. Who creates the solution vector?
#      2. Is there a better way to pass the initial solution information?
default_initializer(problem, t) = error("No default initializer available for a problem of type $(typeof(problem))")

abstract type AbstractProblem end # Temporary helper for CommonSolve.jl until we have finalized the interface

struct NullProblem <: AbstractProblem
    ndofs::Int
end

solution_size(problem::NullProblem) = problem.ndofs

default_initializer(problem::NullProblem, t) = zeros(problem.ndofs)

struct CoupledProblem{MT, CT} <: AbstractProblem
    base_problems::MT
    couplers::CT
end

solution_size(problem::CoupledProblem) = sum([solution_size(p) for p ∈ problem.base_problems])

function default_initializer(problem::CoupledProblem, t)
    mortar([default_initializer(p,t) for p ∈ problem.base_problems])
end

# TODO support arbitrary splits
struct SplitProblem{APT, BPT} <: AbstractProblem
    A::APT
    B::BPT
end

solution_size(problem::SplitProblem) = (solution_size(problem.A), solution_size(problem.B))

default_initializer(problem::SplitProblem, t) = (default_initializer(problem.A, t), default_initializer(problem.B, t))

# TODO support arbitrary partitioning
struct PartitionedProblem{APT, BPT} <: AbstractProblem
    A::APT
    B::BPT
end

solution_size(problem::PartitionedProblem) = solution_size(problem.A) + solution_size(problem.B)

abstract type AbstractPointwiseProblem <: AbstractProblem end

struct ODEProblem{ODET,F,P} <: AbstractProblem
    ode::ODET
    f::F
    p::P
end

solution_size(problem::ODEProblem) = num_states(problem.ode)

function default_initializer(problem::ODEProblem, t) 
    u = zeros(num_states(problem.ode))
    initial_condition!(u, problem.ode)
    u
end

struct PointwiseODEProblem{ODET} <: AbstractPointwiseProblem
    npoints::Int
    ode::ODET
end

solution_size(problem::PointwiseODEProblem) = problem.npoints*num_states(problem.ode)

default_initializer(problem::PointwiseODEProblem, t) = default_initializer(problem.ode, t)

struct TransientHeatProblem{DTF, ST, DH}
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
    function TransientHeatProblem(diffusion_tensor_field::DTF, source_term::ST, dh::DH) where {DTF, ST, DH}
        check_subdomains(dh)
        return new{DTF, ST, DH}(diffusion_tensor_field, source_term, dh)
    end
end

solution_size(problem::TransientHeatProblem) = ndofs(problem.dh)

"""
    QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A discrete problem with time dependent terms and no time derivatives w.r.t. any solution variable.
Abstractly written we want to solve the problem F(u, t) = 0 on some time interval [t₁, t₂].
"""
struct QuasiStaticNonlinearProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH} <: AbstractProblem
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
    function QuasiStaticNonlinearProblem(dh::DH, ch::CH, constitutive_model::CM, face_models::FACE) where {CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH}
        check_subdomains(dh)
        return new{CM, DH, FACE, CH}(dh, ch, constitutive_model, face_models)
    end
end

solution_size(problem::QuasiStaticNonlinearProblem) = ndofs(problem.dh)

default_initializer(problem::QuasiStaticNonlinearProblem, t) = zeros(ndofs(problem.dh))

"""
    QuasiStaticODEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.

TODO implement.
"""
struct QuasiStaticODEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH} <: AbstractProblem
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
struct QuasiStaticDAEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH} <: AbstractProblem
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
end
