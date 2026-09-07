using Test, Thunderbolt, Ferrite, Tensors, LinearAlgebra, Logging

# Rigid rotation of a mesh. Nothing about a long axis may depend on the geometry happening to be
# aligned with the z axis, so every estimator is checked for equivariance under this.
function rotated_mesh(mesh::Thunderbolt.SimpleMesh, R)
    grid = mesh.grid
    nodes = [Node(R ⋅ get_node_coordinate(n)) for n in getnodes(grid)]
    return to_mesh(
        Grid(
            collect(getcells(grid)),
            nodes;
            cellsets = Ferrite.getcellsets(grid),
            nodesets = Ferrite.getnodesets(grid),
            facetsets = Ferrite.getfacetsets(grid),
        ),
    )
end

# `fit_basal_plane` and `compute_principal_axis` return lines, whose sign is arbitrary.
collinear(a, b) = abs(a ⋅ b) ≈ 1

# Independent reimplementation of the reported discrepancies, to check the bookkeeping.
angle_deg(a, b) = rad2deg(acos(clamp(a ⋅ b, -1.0, 1.0)))

@testset "Long axis" begin
    # The idealized LV has its apex at +z and its base clipped flat below the equator, so the long
    # axis is exactly -z and all three estimators have to agree on it.
    lv = generate_ideal_lv_mesh(8, 2, 4)
    ẑ = Vec((0.0, 0.0, 1.0))

    @testset "fit_basal_plane" begin
        normal, centroid, rms = fit_basal_plane(lv)
        @test collinear(normal, ẑ)
        @test rms < 1.0e-12                       # the base is clipped flat
        @test centroid[1] ≈ 0 atol = 1.0e-12      # an annulus centred on the axis
        @test centroid[2] ≈ 0 atol = 1.0e-12
        @test centroid[3] < 0                     # the base sits below the equator

        @testset "in the deformed configuration" begin
            # A rigid translation moves the plane and leaves its normal alone.
            shift = Vec((0.1, 0.2, 0.3))
            n, c, r = fit_basal_plane(lv; u_nodal = fill(shift, getnnodes(lv)))
            @test c ≈ centroid + shift
            @test collinear(n, normal)
            @test r < 1.0e-12

            # Tilting by `z -> z + αx` turns the normal into `(-α, 0, 1)`, exactly: the
            # displacement is linear and the basal facets are bilinear, so nothing is lost.
            α = 0.25
            u = [Vec((0.0, 0.0, α * get_node_coordinate(n)[1])) for n in getnodes(lv)]
            n, _, r = fit_basal_plane(lv; u_nodal = u)
            @test collinear(n, Vec((-α, 0.0, 1.0)) / sqrt(1 + α^2))
            @test r < 1.0e-12                     # a tilted plane is still a plane
        end

        @testset "second_restriction" begin
            base_cells = Set(idx[1] for idx in getfacetset(lv, "Base"))
            posx = Set(c for c in base_cells if sum(getcoordinates(lv.grid, c))[1] > 0)
            @test 0 < length(posx) < length(base_cells)

            n, c, r = fit_basal_plane(lv; second_restriction = posx)
            @test c[1] > 0.4                      # only the +x half of the annulus contributes
            @test c[3] ≈ centroid[3] atol = 1.0e-12
            @test collinear(n, ẑ)                 # a subset of a plane is still that plane
            @test r < 1.0e-12

            @test_throws ArgumentError fit_basal_plane(lv; second_restriction = Set{Int}())
        end
    end

    @testset "compute_principal_axis" begin
        axis, separation = compute_principal_axis(lv)
        @test collinear(axis, ẑ)
        @test separation > 1.0e6                  # strongly transversely isotropic

        # A box with three distinct extents has no degenerate transverse pair, so there is no axis
        # to identify and `separation` has to say so rather than name one confidently.
        box = to_mesh(
            generate_grid(Hexahedron, (3, 3, 3), Vec((0.0, 0.0, 0.0)), Vec((1.0, 2.0, 3.0))),
        )
        @test compute_principal_axis(box)[2] < 2
    end

    @testset "compute_long_axis" begin
        lai = compute_long_axis(lv)
        @test lai.axis_from === :basal_plane
        @test lai.axis ≈ -ẑ                       # apex at +z, base below it
        @test lai.apex ≈ Vec((0.0, 0.0, 1.5))     # the epicardial apex node
        @test lai.base_center[3] ≈ -0.4635254915624207
        @test lai.chamber_length ≈ lai.apex[3] - lai.base_center[3]
        @test lai.base_rms_residual < 1.0e-12

        # On a geometry this symmetric the three estimators have to land on the same axis.
        @test lai.base_normal_discrepancy < 1.0e-6
        @test lai.apex_base_discrepancy < 1.0e-6
        @test lai.inertia_discrepancy < 1.0e-6
        @test lai.inertia_conditioning > 1.0e6

        @testset "$sym selects its own estimator" for (sym, field) in (
            (:basal_plane, :base_normal),
            (:apex_base, :apex_base_axis),
            (:inertia, :inertia_axis),
        )
            l = compute_long_axis(lv; axis_from = sym)
            @test l.axis_from === sym
            @test l.axis == getproperty(l, field)
        end

        @test_throws ArgumentError compute_long_axis(lv; axis_from = :something_else)
    end

    @testset "Estimators that disagree" begin
        # Flattening the septum and stretching one short axis breaks the axisymmetry the inertia
        # estimator needs, and pulls the annotated apex off the axis. The basal plane does not care,
        # which is why it is the default.
        skewed = Thunderbolt.generate_ideal_lv_mesh_hex(8, 2, 4)
        l = compute_long_axis(skewed)

        @test l.axis ≈ -ẑ                          # the base is still clipped flat
        @test l.base_rms_residual < 1.0e-12
        @test l.apex_base_discrepancy > 5           # apex annotation pulled off the axis
        @test l.inertia_discrepancy > 30            # inertia axis is badly wrong here ...
        @test l.inertia_conditioning < 10           # ... and says so

        # Every reported axis is oriented apex -> base, even though the estimators produce lines.
        @test l.base_normal ⋅ l.apex_base_axis > 0
        @test l.inertia_axis ⋅ l.apex_base_axis > 0

        # The discrepancies are angles against the *selected* axis, so they move with `axis_from`.
        @test l.base_normal_discrepancy ≈ angle_deg(l.base_normal, l.axis) atol = 1.0e-9
        @test l.apex_base_discrepancy ≈ angle_deg(l.apex_base_axis, l.axis) atol = 1.0e-9
        @test l.inertia_discrepancy ≈ angle_deg(l.inertia_axis, l.axis) atol = 1.0e-9

        li = compute_long_axis(skewed; axis_from = :inertia)
        @test li.inertia_discrepancy < 1.0e-6                    # zero for the selected one ...
        @test li.base_normal_discrepancy ≈ l.inertia_discrepancy # ... and the roles swap
    end

    @testset "Rotating the geometry rotates the axis" begin
        R = Tensors.rotation_tensor(Vec((1.0, 2.0, 3.0)) / sqrt(14.0), 0.7)
        rotated = rotated_mesh(lv, R)

        @testset "$sym" for sym in (:basal_plane, :apex_base, :inertia)
            a = compute_long_axis(lv; axis_from = sym)
            b = compute_long_axis(rotated; axis_from = sym)
            @test b.axis ≈ R ⋅ a.axis
            @test b.apex ≈ R ⋅ a.apex
            @test b.base_center ≈ R ⋅ a.base_center
            @test b.chamber_length ≈ a.chamber_length
        end

        @test collinear(fit_basal_plane(rotated)[1], R ⋅ fit_basal_plane(lv)[1])
        @test collinear(compute_principal_axis(rotated)[1], R ⋅ compute_principal_axis(lv)[1])
    end

    @testset "Geometry without an apex annotation" begin
        # A ring has no apex, but it has a second planar face opposite the base, whose centroid
        # serves as the far end of the axis.
        ring = generate_ring_mesh(8, 2, 2)
        lai = compute_long_axis(ring; far_facetset = "Myocardium", base_facetset = "Base")
        @test lai.axis ≈ -ẑ
        @test lai.apex[3] ≈ 0.2
        @test lai.base_center[3] ≈ -0.2
        @test lai.chamber_length ≈ 0.4

        # Without one, an empty apex nodeset is an error rather than a meaningless axis.
        grid = generate_grid(Hexahedron, (2, 2, 2))
        with_logger(NullLogger()) do
            addnodeset!(grid, "Empty", x -> false)
        end
        @test_throws ArgumentError compute_long_axis(
            to_mesh(grid);
            apex_nodeset = "Empty",
            base_facetset = "top",
        )
    end

    @testset "show" begin
        out = sprint(show, compute_long_axis(lv))
        @test occursin("basal_plane", out)
        @test occursin("LongAxisInfo", out)
    end
end
