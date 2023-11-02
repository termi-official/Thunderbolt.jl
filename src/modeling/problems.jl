struct NullProblem
    ndofs::Int
end

struct CoupledProblem{MT, CT}
    base_problems::MT
    couplers::CT
end

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

"""
    QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A discrete problem with time dependent terms and no time derivatives w.r.t. any solution variable.
Abstractly written we want to solve the problem F(u, t) = 0 on some time interval [t₁, t₂].
"""
struct QuasiStaticNonlinearProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH, CAL}
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
    calcium_field::CAL
end

"""
    QuasiStaticODEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.

TODO implement.
"""
struct QuasiStaticODEProblem{CM <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler, FACE, CH, CAL, MM}
    dh::DH
    ch::CH
    constitutive_model::CM
    face_models::FACE
    # TODO where to put this?
    microstructure_model::MM
    calcium_field::CAL
end
