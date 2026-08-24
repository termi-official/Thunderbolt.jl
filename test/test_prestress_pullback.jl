using Test
using Thunderbolt
using Thunderbolt:
    material_routine,
    reduced_prestressed_material_routine,
    setup_coefficient_cache,
    setup_internal_cache,
    QuadraturePoint,
    OrthotropicMicrostructure,
    ConstantCoefficient
using Thunderbolt.Tensors
import Ferrite

# The prestressed pullback against automatic differentiation of its defining energy.
#
# Convention under test: the inner model's energy is per unit volume of the *intermediate*
# (stress-free) configuration, so Ψ(F) = det(F₀)·Ψᵉ(F F₀⁻¹) and both P and the tangent carry the
# det(F₀) factor. The AD reference below *is* that definition, so any convention drift in the
# hand-derived pullback — the missing J₀ this test was written against, a transpose, an index
# order — shows up as a mismatch rather than needing a hand-computed expected value.
@testset "prestressed pullback matches AD of det(F₀)·Ψᵉ(F F₀⁻¹)" begin
    grid = Ferrite.generate_grid(Ferrite.Hexahedron, (1, 1, 1))
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :d, Ferrite.Lagrange{Ferrite.RefHexahedron, 1}()^3)
    Ferrite.close!(dh)
    sdh = first(dh.subdofhandlers)
    qr  = Ferrite.QuadratureRule{Ferrite.RefHexahedron}(2)
    cc  = Ferrite.CellCache(grid)
    Ferrite.reinit!(cc, 1)
    qp = QuadraturePoint(1, first(Ferrite.getpoints(qr)))

    ms = OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0)))
    inner = PK1Model(HolzapfelOgden2009Model(), ConstantCoefficient(ms))
    F = one(Tensor{2, 3}) + Tensor{2, 3}((0.05, 0.02, 0.0, -0.01, -0.03, 0.02, 0.01, 0.0, 0.04))

    # Non-isochoric F₀⁻¹ (det ≈ 0.969, the same field the integration tests use) and an isochoric
    # control, normalized so det = 1 exactly up to floating point.
    F₀inv_aniso = Tensor{2, 3}((1.1, 0.1, 0.0, 0.2, 0.9, 0.1, -0.1, 0.0, 1.0))
    F₀inv_iso   = F₀inv_aniso / cbrt(det(F₀inv_aniso))

    for F₀inv in (F₀inv_aniso, F₀inv_iso)
        model = PrestressedMechanicalModel(inner, ConstantCoefficient(F₀inv))
        coeff_cache = setup_coefficient_cache(model, qr, sdh)
        state_cache = setup_internal_cache(model, qr, sdh)

        W(F_) = 1 / det(F₀inv) * Thunderbolt.Ψ(F_ ⋅ F₀inv, ms, HolzapfelOgden2009Model())
        P_ad = Tensors.gradient(W, F)
        A_ad = Tensors.hessian(W, F)

        P, A = material_routine(model, F, coeff_cache, state_cache, cc, qp, 0.0)
        @test P ≈ P_ad
        @test A ≈ A_ad

        P_red =
            reduced_prestressed_material_routine(model, F, coeff_cache, state_cache, cc, qp, 0.0)
        @test P_red ≈ P_ad
    end
end
