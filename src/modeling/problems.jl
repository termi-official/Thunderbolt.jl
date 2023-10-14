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
