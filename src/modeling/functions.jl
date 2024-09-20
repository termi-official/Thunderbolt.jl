# For the mapping against the SciML ecosystem, a "Thunderbolt function" is essentially equivalent to a "SciML function" with parameters, which does not have all evaluation information
"""
    AbstractSemidiscreteFunction <: DiffEqBase.AbstractDiffEqFunction{iip=true}

Supertype for all functions coming from PDE discretizations.

## Interface

    solution_size(::AbstractSemidiscreteFunction)
"""
abstract type AbstractSemidiscreteFunction <: DiffEqBase.AbstractDiffEqFunction{true} end

abstract type AbstractPointwiseFunction <: AbstractSemidiscreteFunction end

"""
    AbstractSemidiscreteBlockedFunction <: AbstractSemidiscreteFunction

Supertype for all functions coming from PDE discretizations with blocked structure.

## Interface

    BlockArrays.blocksizes(::AbstractSemidiscreteFunction)
    BlockArrays.blocks(::AbstractSemidiscreteFunction) -> Iterable
"""
abstract type AbstractSemidiscreteBlockedFunction <: AbstractSemidiscreteFunction end
solution_size(f::AbstractSemidiscreteBlockedFunction) = sum(blocksizes(f))
num_blocks(f::AbstractSemidiscreteBlockedFunction) = length(blocksizes(f))


"""
    NullFunction(ndofs)

Utility type to describe that Jacobian and residual are zero, but ndofs dofs are present.
"""
struct NullFunction <: AbstractSemidiscreteFunction
    ndofs::Int
end

solution_size(f::NullFunction) = f.ndofs


# TODO replace this with the original
struct ODEFunction{ODET,F,P} <: AbstractSemidiscreteFunction
    ode::ODET
    f::F
    p::P
end

solution_size(f::ODEFunction) = num_states(f.ode)

# See https://github.com/JuliaGPU/Adapt.jl/issues/84 for the reason why hardcoding Int does not work
struct PointwiseODEFunction{IndexType <: Integer, ODEType, xType} <: AbstractPointwiseFunction
    npoints::IndexType
    ode::ODEType
    x::xType
end
Adapt.@adapt_structure PointwiseODEFunction

solution_size(f::PointwiseODEFunction) = f.npoints*num_states(f.ode)

struct AffineODEFunction{MI, BI, ST, DH} <: AbstractSemidiscreteFunction
    mass_term::MI
    bilinear_term::BI
    source_term::ST
    dh::DH
end

solution_size(f::AffineODEFunction) = ndofs(f.dh)

struct AffineSteadyStateFunction{BI, ST, DH, CH} <: AbstractSemidiscreteFunction
    bilinear_term::BI
    source_term::ST
    dh::DH
    ch::CH
end

solution_size(f::AffineSteadyStateFunction) = ndofs(f.dh)

abstract type AbstractQuasiStaticFunction <: AbstractSemidiscreteFunction end

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
end

solution_size(f::QuasiStaticNonlinearFunction) = ndofs(f.dh)

"""
    QuasiStaticODEFunction{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.
"""
struct QuasiStaticODEFunction{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, QH, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractQuasiStaticFunction #<: AbstractSemidiscreteODEFunction
    dh::DH
    qh::QH
    ch::CH
    constitutive_model::CM
    face_models::FACE
end

solution_size(f::QuasiStaticODEFunction) = ndofs(f.dh)+ndofs(f.qh)

# """
#     QuasiStaticDAEFunction{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

# A problem with time dependent terms and time derivatives only w.r.t. internal solution variable which can't be expressed as an ODE.

# TODO implement.
# """
# struct QuasiStaticDAEFunction{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractSemidiscreteDAEFunction
#     dh::DH
#     ch::CH
#     constitutive_model::CM
#     face_models::FACE
# end
