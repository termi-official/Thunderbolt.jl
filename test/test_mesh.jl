using Test, Thunderbolt, Tensors
@testset "Mesh" begin
    num_refined_elements(::Type{Hexahedron}) = 8
    num_refined_elements(::Type{Tetrahedron}) = 8
    num_refined_elements(::Type{Triangle}) = 4
    num_refined_elements(::Type{Quadrilateral}) = 4

    function test_detJ(grid)
        for cc ∈ CellIterator(grid)
            cell = getcells(grid, cellid(cc))
            ref_shape = Ferrite.getrefshape(cell)
            ip = getinterpolation(LagrangeCollection{1}(), ref_shape)
            qr = QuadratureRule{ref_shape}([1.0], [Vec(ntuple(_->0.1, Ferrite.getrefdim(cell)))]) # TODO randomize point
            gv = Ferrite.GeometryMapping{1}(Float64, ip, qr)
            x = getcoordinates(cc)
            mapping = Ferrite.calculate_mapping(gv, 1, x)
            J = Ferrite.getjacobian(mapping)
            @test Ferrite.calculate_detJ(J) > 0
        end
    end

    @testset "Cubioidal $element_type" for element_type ∈ [
        Hexahedron,
        Wedge,
        Tetrahedron,
        # Quadrilateral,
        # Triangle
    ]
        dim = Ferrite.getrefdim(element_type)
        grid = generate_grid(element_type, ntuple(_ -> 4, dim))
        addcellset!(grid, "right_cells", x -> x[1] ≥ 0.0)
        addcellset!(grid, "left_cells", x -> x[1] ≤ 0.0)

        if dim == 3
            grid_hex = Thunderbolt.hexahedralize(grid)
            @test all(typeof.(getcells(grid_hex)) .== Hexahedron) # Test if we really hit all elements
            test_detJ(grid_hex) # And for messed up elements

            # Check for correct transfer of facetsets
            addfacetset!(grid_hex, "right_new", x -> x[1] ≈ 1.0)
            @test getfacetset(grid_hex, "right") == getfacetset(grid_hex, "right_new")
            addfacetset!(grid_hex, "left_new", x -> x[1] ≈ -1.0)
            @test getfacetset(grid_hex, "left") == getfacetset(grid_hex, "left_new")
            addfacetset!(grid_hex, "top_new", x -> x[3] ≈ 1.0)
            @test getfacetset(grid_hex, "top") == getfacetset(grid_hex, "top_new")
            addfacetset!(grid_hex, "bottom_new", x -> x[3] ≈ -1.0)
            @test getfacetset(grid_hex, "bottom") == getfacetset(grid_hex, "bottom_new")
            addfacetset!(grid_hex, "front_new", x -> x[2] ≈ -1.0)
            @test getfacetset(grid_hex, "front") == getfacetset(grid_hex, "front_new")
            addfacetset!(grid_hex, "back_new", x -> x[2] ≈ 1.0)
            @test getfacetset(grid_hex, "back") == getfacetset(grid_hex, "back_new")

            # Check for correct transfer of cellsets
            addcellset!(grid_hex, "right_cells_new", x -> x[1] ≥ 0.0)
            @test getcellset(grid_hex, "right_cells") == getcellset(grid_hex, "right_cells_new")
            addcellset!(grid_hex, "left_cells_new", x -> x[1] ≤ 0.0)
            @test getcellset(grid_hex, "left_cells") == getcellset(grid_hex, "left_cells_new")
        end

        if element_type == Hexahedron
            grid_fine = Thunderbolt.uniform_refinement(grid)
            @test getncells(grid_fine) == num_refined_elements(element_type)*getncells(grid)
            @test all(typeof.(getcells(grid_fine)) .== element_type) # For the tested elements all fine elements are the same type
            test_detJ(grid_fine)
        end
    end

    @testset "Linear Hex Ring" begin
        ring_mesh = generate_ring_mesh(8, 3, 3)
        test_detJ(ring_mesh)
        open_ring_mesh = generate_open_ring_mesh(8, 3, 3, π/4)
        test_detJ(open_ring_mesh)
    end

    @testset "Quadratic Hex Ring" begin
        ring_mesh = generate_quadratic_ring_mesh(5, 3, 3)
        test_detJ(ring_mesh)
        open_ring_mesh = generate_quadratic_open_ring_mesh(8, 3, 3, π/4)
        test_detJ(open_ring_mesh)
    end

    @testset "Linear Mixed LV to Hex" begin
        lv_mesh = Thunderbolt.generate_ideal_lv_mesh(8, 4, 4)
        test_detJ(lv_mesh)
        lv_mesh_hex = Thunderbolt.hexahedralize(lv_mesh)
        test_detJ(lv_mesh_hex)
    end

    @testset "Linear Hex LV" begin
        lv_mesh = Thunderbolt.generate_ideal_lv_mesh_hex(8, 4, 4)
        test_detJ(lv_mesh)
        lv_mesh_hex = Thunderbolt.hexahedralize(lv_mesh)
        test_detJ(lv_mesh_hex)
    end

    @testset "Ideal left heart" begin
        # Every facet that belongs to exactly one cell, i.e. the boundary the mesh actually has,
        # derived from the connectivity rather than from what the generator declared.
        function boundary_skeleton(grid)
            owners = Dict{NTuple{4, Int}, Vector{FacetIndex}}()
            for cellid = 1:getncells(grid)
                for (local_facet, facet) in enumerate(Ferrite.facets(getcells(grid, cellid)))
                    # Triangular and quadrilateral facets meet here, so the key is zero padded.
                    sorted = sort!(collect(facet))
                    key = ntuple(i -> i ≤ length(sorted) ? sorted[i] : 0, 4)
                    push!(get!(owners, key, FacetIndex[]), FacetIndex(cellid, local_facet))
                end
            end
            return Set(only(v) for v in values(owners) if length(v) == 1)
        end

        nc, nr, n_lv, n_la = 8, 2, 3, 2
        mesh = generate_ideal_lh_mesh(nc, nr, n_lv, n_la)
        # The annulus plane of the default geometry, where the two shells meet.
        z_rim = 1.5*cos(1.2*π/2)
        test_detJ(mesh)

        @testset "layout" begin
            @test getnnodes(mesh) == nc*(nr+1)*(n_lv+1+n_la) + 2*(nr+1)
            @test getncells(mesh) == nc*nr*(n_lv+n_la) + 2*nc*nr
            @test length(getcellset(mesh, "ventricle")) == nc*nr*(n_lv+1)
            @test length(getcellset(mesh, "atrium")) == nc*nr*(n_la+1)
            @test isempty(intersect(getcellset(mesh, "ventricle"), getcellset(mesh, "atrium")))
            @test length(getcellset(mesh, "ventricle")) + length(getcellset(mesh, "atrium")) ==
                  getncells(mesh)
        end

        @testset "boundary integrity" begin
            lv_endo = getfacetset(mesh, "LVEndocardium")
            la_endo = getfacetset(mesh, "LAEndocardium")
            epi     = getfacetset(mesh, "Epicardium")

            @test !Thunderbolt._has_facetset(mesh, "Base")
            @test isempty(intersect(lv_endo, la_endo))
            @test isempty(intersect(lv_endo, epi))
            @test isempty(intersect(la_endo, epi))
            @test Set(union(lv_endo, la_endo, epi)) == boundary_skeleton(mesh)
            # The annulus is where the two shells meet, so its facets are interior.
            @test isempty(intersect(getfacetset(mesh, "MitralAnnulus"), boundary_skeleton(mesh)))
        end

        @testset "set consistency" begin
            @test getfacetset(mesh, "Endocardium") ==
                  union(getfacetset(mesh, "LVEndocardium"), getfacetset(mesh, "LAEndocardium"))
            @test getfacetset(mesh, "Epicardium") ==
                  union(getfacetset(mesh, "LVEpicardium"), getfacetset(mesh, "LAEpicardium"))
            # `compute_lv_coordinate_system(mesh; subdomains = ["ventricle"])` reads these, so they
            # may not reach into the atrium.
            ventricle = getcellset(mesh, "ventricle")
            for name in ("SRidgePost", "SRidgeAnt", "LVEndocardium", "LVEpicardium", "MitralAnnulus")
                @test all(facet -> facet[1] ∈ ventricle, getfacetset(mesh, name))
            end
            @test all(facet -> facet[1] ∈ getcellset(mesh, "atrium"),
                      getfacetset(mesh, "LAEndocardium"))
        end

        @testset "rim sharing" begin
            rim = getnodeset(mesh, "MitralAnnulus")
            @test length(rim) == nc*(nr+1)

            chambers_of = Dict(n => Set{String}() for n in rim)
            for name in ("ventricle", "atrium"), cellid in getcellset(mesh, name)
                for n in getcells(mesh, cellid).nodes
                    n ∈ rim && push!(chambers_of[n], name)
                end
            end
            @test all(==(Set(["ventricle", "atrium"])), values(chambers_of))

            # Both shells are truncated in the same plane, which is what lets them share the ring.
            @test all(n -> Ferrite.get_node_coordinate(getnodes(mesh, n))[3] ≈ z_rim, rim)
        end

        @testset "chamber separation helper" begin
            endocardium = getfacetset(mesh, "Endocardium")
            plane_point = Vec((0.0, 0.0, z_rim))
            above, below =
                separate_chamber_surfaces(mesh, endocardium, plane_point, Vec((0.0, 0.0, 1.0)))
            @test above == getfacetset(mesh, "LVEndocardium")
            @test below == getfacetset(mesh, "LAEndocardium")

            # Flipping the normal swaps the two returns, which is the documented convention.
            flipped_above, flipped_below =
                separate_chamber_surfaces(mesh, endocardium, plane_point, Vec((0.0, 0.0, -1.0)))
            @test flipped_above == below
            @test flipped_below == above
        end

        @testset "rim reachability" begin
            # The atrium has to be wide enough at the annulus to stand on the ventricular rim.
            @test_throws ErrorException generate_ideal_lh_mesh(4, 1, 1, 1; la_inner_radius = 0.1)
            @test_throws ErrorException generate_ideal_lh_mesh(4, 1, 1, 1; la_outer_radius = 0.1)
        end

        @testset "single chamber generator unchanged" begin
            # The refactor that gave both generators their shared builders may not move the
            # single-chamber mesh: `test/test_rsafdq_operator.jl` pins assembled values on it.
            lv = generate_ideal_lv_mesh(4, 2, 2)
            @test getnnodes(lv) == 4*3*3 + 3
            @test getncells(lv) == 4*2*2 + 4*2
            @test length(getfacetset(lv, "Endocardium")) == 4*2 + 4
            @test length(getfacetset(lv, "Epicardium")) == 4*2 + 4
            @test length(getfacetset(lv, "Base")) == 4*2
            @test length(getfacetset(lv, "SRidgePost")) == 2*2 + 2
            @test length(getfacetset(lv, "SRidgeAnt")) == 2*2 + 2

            θ_first, θ_base = 1.2*π/2/3, 1.2*π/2
            @test Ferrite.get_node_coordinate(getnodes(lv, 1)) ≈
                  Vec((0.7*sin(θ_first), 0.0, 1.3*cos(θ_first)))
            @test Ferrite.get_node_coordinate(getnodes(lv, 2*4*3 + 1)) ≈
                  Vec((0.7*sin(θ_base), 0.0, 1.5*cos(θ_base)))
            @test Ferrite.get_node_coordinate(getnodes(lv, getnnodes(lv))) ≈ Vec((0.0, 0.0, 1.5))
        end
    end

    @testset "IO" begin
        dirname = @__DIR__

        @testset "voom2 legacy" begin
            voom2_grid = load_voom2_grid(dirname * "/data/voom2/ex1")

            @test length(voom2_grid.nodes) == 9
            @test typeof(voom2_grid.cells[1]) == Line
            @test typeof(voom2_grid.cells[2]) == Hexahedron
            @test length(voom2_grid.cells) == 2
            test_detJ(voom2_grid)
        end

        @testset "mfem v1.0 $filename" for (filename, element_type) in [
            ("ref-segment.mesh", Line),
            ("ref-triangle.mesh", Triangle),
            ("ref-square.mesh", Quadrilateral),
            ("ref-tetrahedron.mesh", Tetrahedron),
            ("ref-cube.mesh", Hexahedron),
            ("ref-prism.mesh", Wedge),
            ("ref-pyramid.mesh", Pyramid),
        ]
            mfem_grid = load_mfem_grid(dirname * "/data/mfem/" * filename)

            @test all(typeof.(mfem_grid.cells) .== element_type)
            test_detJ(mfem_grid)
        end

        @testset "openCARP $filename" for (filename, element_type) in [
            ("ref-segment", Line),
            ("ref-triangle", Triangle),
            ("ref-square", Quadrilateral),
            ("ref-tetrahedron", Tetrahedron),
            ("ref-cube", Hexahedron),
            ("ref-prism", Wedge),
            # ("ref-pyramid", Pyramid),
        ]
            carp_grid = load_carp_grid(dirname * "/data/openCARP/" * filename)

            @test all(typeof.(carp_grid.cells) .== element_type)
            test_detJ(carp_grid)
        end
    end

    @testset "Surface extraction" begin
        LV_mesh = generate_ideal_lv_mesh(4, 2, 2)
        surface_mesh = Thunderbolt.extract_outer_surface_mesh(LV_mesh)

        @test length(surface_mesh.cellsets) == 3
        @test length(surface_mesh.cellsets["Epicardium"]) == 3*4
        @test length(surface_mesh.cellsets["Endocardium"]) == 3*4
        @test length(surface_mesh.cellsets["Base"]) == 4*2
        @test length(surface_mesh.nodes) == 2 + 3*2*4 + 4 # nodes at base + nodes on loopes endo&epi + node loop on base
        @test length(surface_mesh.cells) == 2*4 + 2*4*(2+1) # cells at base + cells inside&outside
    end

    @testset "Geometry Tools" begin
        ring_mesh = generate_ring_mesh(5, 4, 4)
        @test Thunderbolt.compute_center_of_mass(ring_mesh) ≈ Vec((0.0, 0.0, 0.0)) atol=1e-16
        @test Thunderbolt.compute_center_of_surface(ring_mesh, "Endocardium") ≈ Vec((0.0, 0.0, 0.0)) atol=1e-16
    end
end
