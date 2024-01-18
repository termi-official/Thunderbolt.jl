@testset "microstructures" begin
    ref_shape = RefHexahedron
    order = 1
    ring_grid = generate_ring_mesh(80,1,1)

    qr_collection = QuadratureRuleCollection(2)
    qr = getquadraturerule(qr_collection, getcells(ring_grid, 1))
    ip_collection = LagrangeCollection{order}()
    
    cv_collection = CellValueCollection(qr_collection, ip_collection^3, ip_collection^3)
    cv_fsn = Thunderbolt.getcellvalues(cv_collection, getcells(ring_grid, 1))

    ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_collection)

    cv_cs = Thunderbolt.create_cellvalues(ring_cs, qr)

    @testset "Midmyocardial coordinate system" begin
        for cellcache in CellIterator(ring_cs.dh)
            reinit!(cv_cs, cellcache)
            dof_indices = celldofs(cellcache)
            for qp in QuadratureIterator(cv_fsn)
                coords = getcoordinates(cellcache)
                sc = spatial_coordinate(cv_fsn, qp, coords)
                transmural = 4*(norm(Vec(sc[1:2]..., 0.))-0.75)
                @test transmural ≈ function_value(cv_cs, qp, ring_cs.u_transmural[dof_indices]) atol=0.01
            end
        end
    end

    @testset "OrthotropicMicrostructureModel" begin
        ms = create_simple_microstructure_model(ring_cs, ip_collection^3, ip_collection^3,
            endo_helix_angle = deg2rad(0.0),
            epi_helix_angle = deg2rad(0.0),
            endo_transversal_angle = 0.0,
            epi_transversal_angle = 0.0,
            sheetlet_pseudo_angle = deg2rad(0)
        )
        for cellcache in CellIterator(ring_grid)
            reinit!(cv_fsn, cellcache)
            for qp in QuadratureIterator(cv_fsn)
                coords = getcoordinates(cellcache)
                sc = spatial_coordinate(cv_fsn, qp, coords)
                # If we set all angles to 0, then the generator on the ring simply generates sheetlets which point in positive z direction, where the normal point radially outwards.
                ndir = Vec(sc[1:2]..., 0.)/norm(sc[1:2])
                sdir = Vec(0., 0., 1.)
                fdir = sdir × ndir
                fsn = evaluate_coefficient(ms, cellcache, qp, 0)
                @test fsn[1] ≈ fdir atol=0.05
                @test fsn[2] ≈ sdir
                @test fsn[3] ≈ ndir atol=0.05
            end
        end
    end
end
