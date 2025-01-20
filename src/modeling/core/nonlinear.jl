@doc raw"""
    NonlinearIntegrator{CoefficientType}

Represents the integrand a the nonlinear form over some function space.
"""
struct NonlinearIntegrator{VM, FM, QRC, FQRC}
    volume_model::VM
    face_model::FM
    syms::Vector{Symbol}
    qrc::QRC
    fqrc::FQRC
end
