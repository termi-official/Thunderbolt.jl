struct SplitProblem{APT, BPT}
    A::APT
    B::BPT
end

struct PartitionedProblem{APT, BPT}
    A::APT
    B::BPT
end

abstract type AbstractPointwiseProblem end

struct PointwiseODEProblem{ODET} <: AbstractPointwiseProblem
    npoints::Int
    ode::ODET
end

struct TransientHeatProblem{DTF, ST, DH}
    diffusion_tensor_field::DTF
    source_term::ST
    dh::DH
end

# import Thunderbolt: QuasiStaticModel
# """
#     QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

# A discrete problem with time dependent terms and no time derivatives w.r.t. any solution variable.
# Abstractly written we want to solve the problem F(u, t) = 0 on some time interval [t₁, t₂].
# """
# struct QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}
#     model::M
#     dh::DH
# end

# """
#     QuasiStaticDAEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

# A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.
# """
# struct QuasiStaticDAEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}
#     model::M
#     dh::DH
# end
