using Test, Thunderbolt, Tensors, LinearAlgebra
using JET: @test_opt, @test_call
@testset "Element API" begin
    import Thunderbolt: assemble_element!, assemble_facet!
    import Thunderbolt: setup_element_cache, setup_boundary_cache
    import Thunderbolt: BilinearMassIntegrator, BilinearDiffusionIntegrator
    import FerriteOperators

    setup_test_cache(kwargs...) =
        FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
    function setup_test_composite_volume_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeVolumetricElementCache((element_cache, element_cache)),
        )
    end
    function setup_test_composite_surface_cache(kwargs...)
        element_cache = FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            setup_boundary_cache(kwargs...),
        )
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeSurfaceElementCache((element_cache, element_cache)),
        )
    end

    grid = generate_grid(Hexahedron, (1, 1, 1))
    qrc  = QuadratureRuleCollection(3)
    qr   = QuadratureRule{RefHexahedron}(3)
    qrcf = QuadratureRuleCollection(3)
    qrf  = FacetQuadratureRule{RefHexahedron}(3)
    ip   = Lagrange{RefHexahedron, 1}()

    dhs = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    Ferrite.reinit!(cell_cache_s, 1)
    uₑs = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0] .* 1e-4

    ipv = ip^3
    dhv = DofHandler(grid)
    add!(dhv, :u, ipv)
    close!(dhv)
    sdhv = first(dhv.subdofhandlers)
    cell_cache_v = Ferrite.CellCache(sdhv)
    Ferrite.reinit!(cell_cache_v, 1)
    uₑv =
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ] .* 1e-4

    # No we check some examples for the implemented physics
    @testset "Scalar volumetric bilinear elements: $model" for model in (
        BilinearMassIntegrator(ConstantCoefficient(1.0), qrc, :u),
        BilinearDiffusionIntegrator(ConstantCoefficient(one(Tensor{2, 3})), qrc, :u),
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        assemble_element!(Kₑ¹, cell_cache_s, element_cache, 0.0)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_element!(Kₑ², cell_cache_s, composite_element_cache, 0.0)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Vectorial volumetric nonlinear elements: $model" for model in (
        PK1Model(
            HolzapfelOgden2009Model(),
            ConstantCoefficient(
                OrthotropicMicrostructure(
                    Vec((1.0, 0.0, 0.0)),
                    Vec((0.0, 1.0, 0.0)),
                    Vec((0.0, 0.0, 1.0)),
                ),
            ),
        ),
    )
        rₑ¹ = zeros(ndofs(dhv))
        rₑ² = zeros(ndofs(dhv))
        Kₑ¹ = zeros(ndofs(dhv), ndofs(dhv))
        Kₑ² = zeros(ndofs(dhv), ndofs(dhv))

        element_cache = setup_test_cache(QuasiStaticModel(:u, model, ()), qr, sdhv)

        @test_opt assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, element_cache, 0.0)
        assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, element_cache, 0.0)
        @test !iszero(Kₑ¹)
        @test !iszero(rₑ¹)

        @test_opt assemble_element!(rₑ², uₑv, cell_cache_v, element_cache, 0.0)
        assemble_element!(rₑ², uₑv, cell_cache_v, element_cache, 0.0)
        @test rₑ² ≈ rₑ¹

        @test_opt assemble_element!(Kₑ², uₑv, cell_cache_v, element_cache, 0.0)
        assemble_element!(Kₑ², uₑv, cell_cache_v, element_cache, 0.0)
        @test Kₑ² ≈ Kₑ¹

        composite_element_cache =
            setup_test_composite_volume_cache(QuasiStaticModel(:u, model, ()), qr, sdhv)

        Kₑ¹ .= 0.0
        rₑ¹ .= 0.0
        @test_opt assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, composite_element_cache, 0.0)
        assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, composite_element_cache, 0.0)
        @test 2Kₑ² ≈ Kₑ¹
        @test 2rₑ² ≈ rₑ¹

        rₑ² .= 0.0
        @test_opt assemble_element!(rₑ², uₑv, cell_cache_v, composite_element_cache, 0.0)
        assemble_element!(rₑ², uₑv, cell_cache_v, composite_element_cache, 0.0)
        @test rₑ² ≈ rₑ¹

        Kₑ² .= 0.0
        @test_opt assemble_element!(Kₑ², uₑv, cell_cache_v, composite_element_cache, 0.0)
        assemble_element!(Kₑ², uₑv, cell_cache_v, composite_element_cache, 0.0)
        @test Kₑ² ≈ Kₑ¹
    end

    # No we check some examples for the implemented physics
    @testset "Vectorial surface elements: $model" for (model, has_jac) in (
        (RobinBC(1.0, "left"), true),
        (NormalSpringBC(1.0, "left"), true),
        (BendingSpringBC(1.0, "left"), true),
        (ConstantPressureBC(1.0, "left"), true),
        (PressureFieldBC(ConstantCoefficient(1.0), "left"), true),
    )
        rₑ¹ = zeros(ndofs(dhv))
        rₑ² = zeros(ndofs(dhv))
        Kₑ¹ = zeros(ndofs(dhv), ndofs(dhv))
        Kₑ² = zeros(ndofs(dhv), ndofs(dhv))

        element_cache = setup_boundary_cache(model, qrf, sdhv)

        for local_facet_index = 1:nfacets(cell_cache_v)
            assemble_facet!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, local_facet_index, element_cache, 0.0)
            @test iszero(Kₑ¹) != has_jac
            @test iszero(rₑ¹) != has_jac

            assemble_facet!(rₑ², uₑv, cell_cache_v, local_facet_index, element_cache, 0.0)
            @test rₑ² ≈ rₑ¹

            assemble_facet!(Kₑ², uₑv, cell_cache_v, local_facet_index, element_cache, 0.0)
            @test Kₑ² ≈ Kₑ¹
        end

        composite_element_cache = setup_test_composite_surface_cache(model, qrf, sdhv)

        Kₑ¹ .= 0.0
        rₑ¹ .= 0.0
        for local_facet_index = 1:nfacets(cell_cache_v)
            assemble_facet!(
                Kₑ¹,
                rₑ¹,
                uₑv,
                cell_cache_v,
                local_facet_index,
                composite_element_cache,
                0.0,
            )
        end
        @test 2Kₑ² ≈ Kₑ¹
        @test 2rₑ² ≈ rₑ¹

        Kₑ² .= 0.0
        rₑ² .= 0.0
        for local_facet_index = 1:nfacets(cell_cache_v)
            assemble_facet!(rₑ², uₑv, cell_cache_v, local_facet_index, composite_element_cache, 0.0)
            assemble_facet!(Kₑ², uₑv, cell_cache_v, local_facet_index, composite_element_cache, 0.0)
        end
        @test Kₑ² ≈ Kₑ¹
        @test rₑ² ≈ rₑ¹
    end

    # The dashpots are functions of `(u, v)` rather than `(u, t)`, so they are driven through the
    # time scheme's parameter object rather than a bare time. `assemble_element!` is therefore the
    # entry point under test -- it is where the payload is translated into a velocity.
    @testset "Viscous surface elements: $model" for model in (
        ViscousRobinBC(3.0, "left"),
        NormalViscousSpringBC(3.0, "left"),
    )
        n     = ndofs(dhv)
        Δt    = 0.25
        uprev = uₑv ./ 3
        pfot  = FerriteOperators.GenericFirstOrderTimeElementParameters(nothing, 0.0, Δt, uprev)

        cache = setup_boundary_cache(model, qrf, sdhv)

        K¹ = zeros(n, n);
        r¹ = zeros(n)
        FerriteOperators.assemble_element!(K¹, r¹, uₑv, cell_cache_v, cache, pfot)
        @test !iszero(K¹)
        @test !iszero(r¹)

        # The three assembly variants must agree.
        K² = zeros(n, n);
        r² = zeros(n)
        FerriteOperators.assemble_element!(r², uₑv, cell_cache_v, cache, pfot)
        FerriteOperators.assemble_element!(K², uₑv, cell_cache_v, cache, pfot)
        @test r² ≈ r¹
        @test K² ≈ K¹

        # No velocity, no traction -- and the tangent of a linear damper does not care about `u`.
        r⁰ = zeros(n);
        K⁰ = zeros(n, n)
        FerriteOperators.assemble_element!(K⁰, r⁰, uprev, cell_cache_v, cache, pfot)
        @test norm(r⁰) < 1e-14
        @test K⁰ ≈ K¹

        # The tangent is the derivative of the residual. This is the property that a hand written
        # velocity reconstruction gets wrong, so it is checked rather than assumed.
        Kfd = zeros(n, n)
        h = 1e-7
        for i = 1:n
            up = copy(uₑv);
            up[i] += h
            um = copy(uₑv);
            um[i] -= h
            rp = zeros(n);
            rm = zeros(n)
            FerriteOperators.assemble_element!(rp, up, cell_cache_v, cache, pfot)
            FerriteOperators.assemble_element!(rm, um, cell_cache_v, cache, pfot)
            Kfd[:, i] .= (rp .- rm) ./ (2h)
        end
        @test maximum(abs, Kfd .- K¹) < 1e-6 * max(1.0, maximum(abs, K¹))

        # Halving the timestep doubles the velocity, hence both the traction and the tangent.
        Kh = zeros(n, n);
        rh = zeros(n)
        pfoth = FerriteOperators.GenericFirstOrderTimeElementParameters(nothing, 0.0, Δt/2, uprev)
        FerriteOperators.assemble_element!(Kh, rh, uₑv, cell_cache_v, cache, pfoth)
        @test Kh ≈ 2 .* K¹
        @test rh ≈ 2 .* r¹
    end

    @testset "Mixed spring/dashpot boundary" begin
        # Regression test for the element interface: FerriteOperators computes one element parameter
        # object from the *volumetric* cache and hands it to the boundary cache too, so a composite
        # mixing a spring (which wants the time) with a dashpot (which wants the velocity) has to
        # route that one object two different ways. Stripping it to the time at the top -- as the
        # boundary unwrapping used to do unconditionally -- makes the dashpot silently unassemblable.
        n     = ndofs(dhv)
        uprev = uₑv ./ 3
        pfot  = FerriteOperators.GenericFirstOrderTimeElementParameters(nothing, 0.0, 0.25, uprev)

        spring = setup_boundary_cache(NormalSpringBC(5.0, "left"), qrf, sdhv)
        dashpot = setup_boundary_cache(ViscousRobinBC(3.0, "left"), qrf, sdhv)
        composite = FerriteOperators.CompositeSurfaceElementCache((spring, dashpot))

        rs = zeros(n);
        rd = zeros(n);
        rc = zeros(n)
        FerriteOperators.assemble_element!(rs, uₑv, cell_cache_v, spring, pfot)
        FerriteOperators.assemble_element!(rd, uₑv, cell_cache_v, dashpot, pfot)
        FerriteOperators.assemble_element!(rc, uₑv, cell_cache_v, composite, pfot)
        @test !iszero(rs)
        @test !iszero(rd)
        @test rc ≈ rs .+ rd

        Ks = zeros(n, n);
        Kd = zeros(n, n);
        Kc = zeros(n, n)
        FerriteOperators.assemble_element!(Ks, uₑv, cell_cache_v, spring, pfot)
        FerriteOperators.assemble_element!(Kd, uₑv, cell_cache_v, dashpot, pfot)
        FerriteOperators.assemble_element!(Kc, uₑv, cell_cache_v, composite, pfot)
        @test Kc ≈ Ks .+ Kd

        Kc² = zeros(n, n);
        rc² = zeros(n)
        FerriteOperators.assemble_element!(Kc², rc², uₑv, cell_cache_v, composite, pfot)
        @test Kc² ≈ Kc
        @test rc² ≈ rc

        # A spring must see exactly the same thing whether it is handed the payload or a bare time.
        r_time = zeros(n)
        FerriteOperators.assemble_element!(r_time, uₑv, cell_cache_v, spring, 0.0)
        @test rs ≈ r_time
    end
end
