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
num_blocks(::AbstractSemidiscreteBlockedFunction) = length(blocksizes)


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
struct PointwiseODEFunction{IndexType <: Integer, ODEType} <: AbstractPointwiseFunction
    npoints::IndexType
    ode::ODEType
end

solution_size(f::PointwiseODEFunction) = f.npoints*num_states(f.ode)

# TODO translate into AffineODEFunction and use ODEFunction
struct TransientDiffusionFunction{DTF, ST, DH} <: AbstractSemidiscreteFunction
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
end

solution_size(f::TransientDiffusionFunction) = ndofs(f.dh)

struct SteadyDiffusionFunction{DTF, ST, DH, CH} <: AbstractSemidiscreteFunction
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
    ch::CH
end

solution_size(f::SteadyDiffusionFunction) = ndofs(f.dh)

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

# """
#     QuasiStaticODEFunction{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

# A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.

# TODO implement.
# """
# struct QuasiStaticODEFunction{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE <: Tuple, CH <: ConstraintHandler} <: AbstractSemidiscreteODEFunction
#     dh::DH
#     ch::CH
#     constitutive_model::CM
#     face_models::FACE
# end

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
