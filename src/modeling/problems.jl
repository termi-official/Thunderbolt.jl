# For the mapping against the SciML ecosystem, a "Thunderbolt problem" is essentially equivalent to a "SciML function" coupled to a specific "SciML problem".
"""
AbstractSemidiscreteFunction <: DiffEqBase.AbstractDiffEqFunction{iip=true}

Supertype for all functions coming from PDE discretizations.

## Interface

solution_size(::AbstractSemidiscreteFunction)
"""
abstract type AbstractSemidiscreteFunction <: DiffEqBase.AbstractDiffEqFunction{true} end

"""
    AbstractSemidiscreteProblem <: DiffEqBase.AbstractDEProblem

Supertype for all functions coming from PDE discretizations.

## Interface

solution_size(::AbstractSemidiscreteODEProblem)
"""
abstract type AbstractSemidiscreteProblem <: DiffEqBase.AbstractDEProblem end
DiffEqBase.has_kwargs(::AbstractSemidiscreteProblem) = false
DiffEqBase.isinplace(::AbstractSemidiscreteProblem) = true
solution_size(prob::AbstractSemidiscreteProblem) = solution_size(prob.f)

"""
    AbstractSemidiscreteODEProblem <: AbstractSemidiscreteProblem

Supertype for discretizations of time-dependent PDEs in residual form.
"""
abstract type AbstractSemidiscreteDAEProblem <: AbstractSemidiscreteProblem end


"""
    AbstractSemidiscreteODEProblem <: AbstractSemidiscreteProblem

Supertype for discretizations of time-dependent PDEs in mass matrix form.
"""
abstract type AbstractSemidiscreteODEProblem <: AbstractSemidiscreteProblem end

"""
    AbstractSemidiscreteNonlinearProblem <: AbstractSemidiscreteProblem

Supertype for discretizations of time-dependent PDEs without explicit time derivatives.
"""
abstract type AbstractSemidiscreteNonlinearProblem <: AbstractSemidiscreteProblem end

abstract type AbstractPointwiseProblem <: AbstractSemidiscreteProblem end

default_initializer(problem::AbstractSemidiscreteODEProblem, t) = zeros(solution_size(problem))
default_initializer(problem::RSAFDQ2022TyingProblem, t) = zeros(solution_size(problem)) # FIXME


"""
    NullProblem(ndofs)

Utility type to describe that Jacobian and residual are zero, but ndofs dofs are present.
This is a trick for Lagrange-type couplings to describe the Lagrange dofs.
"""
struct NullProblem <: AbstractSemidiscreteProblem
    ndofs::Int
end

solution_size(problem::NullProblem) = problem.ndofs

abstract type AbstractCoupledProblem <: AbstractSemidiscreteProblem end

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


# TODO replace with OS module
struct SplitProblem{APT <: AbstractSemidiscreteODEProblem, BPT <: AbstractSemidiscreteODEProblem} <: AbstractSemidiscreteODEProblem
    A::APT
    B::BPT
end

solution_size(problem::SplitProblem) = (solution_size(problem.A), solution_size(problem.B))

default_initializer(problem::SplitProblem, t) = (default_initializer(problem.A, t), default_initializer(problem.B, t))



# TODO support arbitrary partitioning
struct PartitionedProblem{APT <: AbstractSemidiscreteODEProblem, BPT <: AbstractSemidiscreteODEProblem} <: AbstractSemidiscreteODEProblem
    A::APT
    B::BPT
end

solution_size(problem::PartitionedProblem) = solution_size(problem.A) + solution_size(problem.B)

# TODO replace this with the original
struct ODEProblem{ODET,F,P} <: AbstractSemidiscreteODEProblem
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

# TODO translate into AffineODEFunction and use ODEProblem
struct TransientHeatProblem{DTF, ST, DH} <: AbstractSemidiscreteODEProblem
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
    function TransientHeatProblem(diffusion_tensor_field::DTF, source_term::ST, dh::DH) where {DTF, ST, DH}
        check_subdomains(dh)
        return new{DTF, ST, DH}(diffusion_tensor_field, source_term, dh)
    end
end

solution_size(problem::TransientHeatProblem) = ndofs(problem.dh)

abstract type AbstractQuasiStaticFunction <: AbstractSemidiscreteFunction end

struct QuasiStaticProblem{fType <: AbstractQuasiStaticFunction, uType, tType, pType} <: AbstractSemidiscreteProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

QuasiStaticProblem(f::AbstractQuasiStaticFunction, tspan::Tuple{<:Real, <:Real}) = QuasiStaticProblem(f, zeros(ndofs(f.dh)), tspan, DiffEqBase.NullParameters())

function DiffEqBase.build_solution(prob::AbstractSemidiscreteProblem,
        alg, t, u; timeseries_errors = length(u) > 2,
        dense = false, dense_errors = dense,
        calculate_error = true,
        k = nothing,
        alg_choice = nothing,
        interp = DiffEqBase.LinearInterpolation(t, u),
        retcode = DiffEqBase.ReturnCode.Default, destats = missing, stats = nothing,
        kwargs...)
    T = eltype(eltype(u))
    N = 2 # Why?

    resid = nothing
    original = nothing

    return DiffEqBase.SciMLBase.ODESolution{T, N}(u,
        nothing,
        nothing,
        t, k,
        prob,
        alg,
        interp,
        dense,
        0,
        stats,
        alg_choice,
        retcode,
        resid,
        original)

    # xref https://github.com/xtalax/MethodOfLines.jl/blob/bc0bf8c4fcd2376dc5c3df9642806749bc0c1cdd/src/interface/solution/timedep.jl#L11
    # original_sol = u
    # TODO custom AbstractPDETimeSeriesSolution
    # return DiffEqBase.SciMLBase.PDETimeSeriesSolution{T, N, typeof(u), Nothing, typeof(u)}(u,
    #     original_sol,
    #     nothing,
    #     t, k,
    #     nothing, # ivdomain
    #     nothing, # ivs
    #     nothing, # dvs
    #     nothing, # disc_data
    #     prob,
    #     alg,
    #     interp,
    #     dense,
    #     0,
    #     stats,
    #     retcode,
    #     stats)
end

"""
    QuasiStaticNonlinearFunction{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A discrete problem with time dependent terms and no time derivatives w.r.t. any solution variable.
Abstractly written we want to solve the problem F(u, t) = 0 on some time interval [t₁, t₂].
"""
struct QuasiStaticNonlinearFunction{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractQuasiStaticFunction
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
    function QuasiStaticNonlinearFunction(dh::DH, ch::CH, constitutive_model::CM, face_models::FACE) where {CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH}
        check_subdomains(dh)
        return new{CM, DH, FACE, CH}(dh, ch, constitutive_model, face_models)
    end
end

solution_size(problem::QuasiStaticNonlinearFunction) = ndofs(problem.dh)

# """
#     QuasiStaticODEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

# A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.

# TODO implement.
# """
# struct QuasiStaticODEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractSemidiscreteODEProblem
#     dh::DH
#     ch::CH
#     constitutive_model::CM
#     face_models::FACE
# end

# """
#     QuasiStaticDAEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

# A problem with time dependent terms and time derivatives only w.r.t. internal solution variable which can't be expressed as an ODE.

# TODO implement.
# """
# struct QuasiStaticDAEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractSemidiscreteDAEProblem
#     dh::DH
#     ch::CH
#     constitutive_model::CM
#     face_models::FACE
# end

"""
    RSAFDQ20223DProblem{MT, CT}

Generic description of the problem associated with the RSAFDQModel.
"""
struct RSAFDQ20223DProblem{MT <: QuasiStaticNonlinearFunction, TP <: RSAFDQ2022TyingProblem} <: AbstractCoupledProblem
    structural_problem::MT
    tying_problem::TP
end

base_problems(problem::RSAFDQ20223DProblem) = (problem.structural_problem, problem.tying_problem)




getch(problem) = problem.ch
getch(problem::RSAFDQ20223DProblem) = getch(problem.structural_problem)
