@testset "microstructures" begin
    ref_shape = RefHexahedron
    order = 1

    ip_fsn = Lagrange{ref_shape, order}()^3
    ip_u = Lagrange{ref_shape, order}()^3
    ip_geo = Lagrange{ref_shape, order}()^3
    cv = CellValues(QuadratureRule{ref_shape}(1), ip_fsn, ip_geo)

    ring_grid = generate_ring_mesh(8,2,2)
    ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
    @testset "OrthotropicMicrostructureModel" begin
        ms = create_simple_microstructure_model(ring_cs, ip_fsn, ip_geo,
            endo_helix_angle = deg2rad(0.0),
            epi_helix_angle = deg2rad(0.0),
            endo_transversal_angle = 0.0,
            epi_transversal_angle = 0.0,
            sheetlet_pseudo_angle = deg2rad(0)
        )
        for cellcache in CellIterator(ring_grid)
            reinit!(cv, cellcache)
            for qp in QuadratureIterator(cv)
                coords = getcoordinates(cellcache)
                sc = spatial_coordinate(cv, qp, coords)
                ndir = Vec(sc[1:2]..., 0.)/norm(sc[1:2])
                sdir = Vec(0., 0., 1.)
                fdir = sdir × ndir
                eval_cooef = evaluate_coefficient(ms, cellcache, qp, 0)
                fdir_eval = eval_cooef[1]
                sdir_eval = eval_cooef[2]
                ndir_eval = eval_cooef[3]
                @test fdir_eval ≈ fdir
                @test sdir_eval ≈ sdir
                @test ndir_eval ≈ ndir
            end
        end
    end
end
