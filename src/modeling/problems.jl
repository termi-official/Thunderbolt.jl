# For the mapping against the SciML ecosystem, a "Thunderbolt problem" is essentially equivalent to a "SciML function" coupled to a specific "SciML problem".

# TODO rethink interface here
#      1. Who creates the solution vector?
#      2. Is there a better way to pass the initial solution information?
abstract type AbstractProblem end # Temporary helper for CommonSolve.jl until we have finalized the interface

abstract type AbstractPointwiseProblem <: AbstractProblem end

default_initializer(problem::AbstractProblem, t) = zeros(solution_size(problem))
default_initializer(problem::RSAFDQ2022TyingProblem, t) = zeros(solution_size(problem)) # FIXME


"""
    NullProblem(ndofs)

Utility type to describe that Jacobian and residual are zero, but ndofs dofs are present.
This is a trick for Lagrange-type couplings to describe the Lagrange dofs.
"""
struct NullProblem <: AbstractProblem
    ndofs::Int
end

solution_size(problem::NullProblem) = problem.ndofs

abstract type AbstractCoupledProblem <: AbstractProblem end

"""
    CoupledProblem{MT, CT}

Generic description of a coupled problem.
"""
struct CoupledProblem{MT <: Tuple, CT <: Tuple} <: AbstractCoupledProblem
    base_problems::MT
    couplings::CT
end

base_problems(problem::CoupledProblem) = problem.base_problems

solution_size(problem::AbstractCoupledProblem) = sum([solution_size(p) for p ∈ base_problems(problem)])

function default_initializer(problem::AbstractCoupledProblem, t)
    mortar([default_initializer(p,t) for p ∈ base_problems(problem)])
end

function get_coupler(problem::CoupledProblem, i::Int, j::Int)
    for coupling in problem.couplers
        @unpack coupler = coupling
        is_correct_coupler(coupling.coupler, i, j) && return
    end
    return NullCoupler()
end

relevant_couplings(problem::CoupledProblem, i::Int) = [coupling for coupling in problem.couplings if is_relevant_coupling(coupling)]


# TODO support arbitrary splits
struct SplitProblem{APT <: AbstractProblem, BPT <: AbstractProblem} <: AbstractProblem
    A::APT
    B::BPT
end

solution_size(problem::SplitProblem) = (solution_size(problem.A), solution_size(problem.B))

default_initializer(problem::SplitProblem, t) = (default_initializer(problem.A, t), default_initializer(problem.B, t))



# TODO support arbitrary partitioning
struct PartitionedProblem{APT <: AbstractProblem, BPT <: AbstractProblem} <: AbstractProblem
    A::APT
    B::BPT
end

solution_size(problem::PartitionedProblem) = solution_size(problem.A) + solution_size(problem.B)


struct ODEProblem{ODET,F,P} <: AbstractProblem
    ode::ODET
    f::F
    p::P
end

solution_size(problem::ODEProblem) = num_states(problem.ode)

function default_initializer(problem::ODEProblem, t) 
    u = zeros(num_states(problem.ode))
    default_initial_condition!(u, problem.ode)
    u
end

struct PointwiseODEProblem{ODET} <: AbstractPointwiseProblem
    npoints::Int
    ode::ODET
end

solution_size(problem::PointwiseODEProblem) = problem.npoints*num_states(problem.ode)

default_initializer(problem::PointwiseODEProblem, t) = default_initializer(problem.ode, t)

struct TransientHeatProblem{DTF, ST, DH} <: AbstractProblem
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
struct QuasiStaticNonlinearProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractProblem
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
struct QuasiStaticODEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractProblem
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
struct QuasiStaticDAEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractProblem
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
end

"""
    RSAFDQ20223DProblem{MT, CT}

Generic description of the problem associated with the RSAFDQModel.
"""
struct RSAFDQ20223DProblem{MT <: QuasiStaticNonlinearProblem, TP <: RSAFDQ2022TyingProblem} <: AbstractCoupledProblem
    structural_problem::MT
    tying_problem::TP
end

base_problems(problem::RSAFDQ20223DProblem) = (problem.structural_problem, problem.tying_problem)




getch(problem) = problem.ch
getch(problem::RSAFDQ20223DProblem) = getch(problem.structural_problem)
