@testset "Kopp Quadrilateral Refinement" begin
    ip = DiscontinuousLagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(1);
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(1);
    kopp_values = KoppValues(
        CellValues(qr, ip),
        FacetValues(qr_facet, ip),
        InterfaceValues(qr_facet, ip)
    );
    @testset "3x3 Quadrilateral corner" begin
        grid = generate_grid(KoppCell{2, Int}, (3,3))
        topology = KoppTopology(grid)
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh);
        refinement_cache = KoppRefinementCache(topology);
        kopp_cache = KoppCache(grid, dh, kopp_values, refinement_cache, topology);
        cells_to_refine = Set([CellIndex(1)]);
        refine!(grid, topology, refinement_cache, kopp_cache, kopp_values, cells_to_refine, dh)
        @testset "one level" begin
            @testset "coords" begin    
                coords = zeros(Vec{2, Float64}, 4)
                Ferrite.getcoordinates!(coords, grid, CellIndex(2))
                @test coords ≈ [Vec(-1.0, -1.0), Vec(-4/6, -1.0), Vec(-4/6, -4/6), Vec(-1.0, -4/6)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(3))
                @test coords ≈ [Vec(-4/6, -1.0), Vec(-2/6, -1.0), Vec(-2/6, -4/6), Vec(-4/6, -4/6)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(4))
                @test coords ≈ [Vec(-4/6, -4/6), Vec(-2/6, -4/6), Vec(-2/6, -2/6), Vec(-4/6, -2/6)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(5))
                @test coords ≈ [Vec(-1.0, -4/6), Vec(-4/6, -4/6), Vec(-4/6, -2/6), Vec(-1.0, -2/6)]
            end
            @testset "topology" begin
                @test getneighborhood(topology, grid, FacetIndex(2,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(2,2)) == [FacetIndex(3,4)]
                @test getneighborhood(topology, grid, FacetIndex(2,3)) == [FacetIndex(5,1)]
                @test getneighborhood(topology, grid, FacetIndex(2,4)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(3,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(3,2)) == [FacetIndex(6,4)]
                @test getneighborhood(topology, grid, FacetIndex(3,3)) == [FacetIndex(4,1)]
                @test getneighborhood(topology, grid, FacetIndex(3,4)) == [FacetIndex(2,2)]
                @test getneighborhood(topology, grid, FacetIndex(4,1)) == [FacetIndex(3,3)]
                @test getneighborhood(topology, grid, FacetIndex(4,2)) == [FacetIndex(6,4)]
                @test getneighborhood(topology, grid, FacetIndex(4,3)) == [FacetIndex(8,1)]
                @test getneighborhood(topology, grid, FacetIndex(4,4)) == [FacetIndex(5,2)]
                @test getneighborhood(topology, grid, FacetIndex(5,1)) == [FacetIndex(2,3)]
                @test getneighborhood(topology, grid, FacetIndex(5,2)) == [FacetIndex(4,4)]
                @test getneighborhood(topology, grid, FacetIndex(5,3)) == [FacetIndex(8,1)]
                @test getneighborhood(topology, grid, FacetIndex(5,4)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(6,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(6,2)) == [FacetIndex(7,4)]
                @test getneighborhood(topology, grid, FacetIndex(6,3)) == [FacetIndex(9,1)]
                @test getneighborhood(topology, grid, FacetIndex(6,4)) == [FacetIndex(3,2), FacetIndex(4,2)]
            end
            @testset "two levels" begin
                cells_to_refine = Set([CellIndex(4)]);
                refine!(grid, topology, refinement_cache, kopp_cache, kopp_values, cells_to_refine, dh)
                @testset "coords" begin
                    coords = zeros(Vec{2, Float64}, 4)
                    Ferrite.getcoordinates!(coords, grid, CellIndex(5))
                    @test coords ≈ [Vec(-4/6, -4/6), Vec(-3/6, -4/6), Vec(-3/6, -3/6), Vec(-4/6, -3/6)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(6))
                    @test coords ≈ [Vec(-3/6, -4/6), Vec(-2/6, -4/6), Vec(-2/6, -3/6), Vec(-3/6, -3/6)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(7))
                    @test coords ≈ [Vec(-3/6, -3/6), Vec(-2/6, -3/6), Vec(-2/6, -2/6), Vec(-3/6, -2/6)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(8))
                    @test coords ≈ [Vec(-4/6, -3/6), Vec(-3/6, -3/6), Vec(-3/6, -2/6), Vec(-4/6, -2/6)]
                end
                @testset "topology" begin
                    @test getneighborhood(topology, grid, FacetIndex(2,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(2,2)) == [FacetIndex(3,4)]
                    @test getneighborhood(topology, grid, FacetIndex(2,3)) == [FacetIndex(9,1)]
                    @test getneighborhood(topology, grid, FacetIndex(2,4)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(3,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(3,2)) == [FacetIndex(10,4)]
                    @test getneighborhood(topology, grid, FacetIndex(3,3)) == [FacetIndex(5,1), FacetIndex(6,1)]
                    @test getneighborhood(topology, grid, FacetIndex(3,4)) == [FacetIndex(2,2)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(5,1)) == [FacetIndex(3,3)]
                    @test getneighborhood(topology, grid, FacetIndex(5,2)) == [FacetIndex(6,4)]
                    @test getneighborhood(topology, grid, FacetIndex(5,3)) == [FacetIndex(8,1)]
                    @test getneighborhood(topology, grid, FacetIndex(5,4)) == [FacetIndex(9,2)]
                    @test getneighborhood(topology, grid, FacetIndex(6,1)) == [FacetIndex(3,3)]
                    @test getneighborhood(topology, grid, FacetIndex(6,2)) == [FacetIndex(10,4)]
                    @test getneighborhood(topology, grid, FacetIndex(6,3)) == [FacetIndex(7,1)]
                    @test getneighborhood(topology, grid, FacetIndex(6,4)) == [FacetIndex(5,2)]
                    @test getneighborhood(topology, grid, FacetIndex(7,1)) == [FacetIndex(6,3)]
                    @test getneighborhood(topology, grid, FacetIndex(7,2)) == [FacetIndex(10,4)]
                    @test getneighborhood(topology, grid, FacetIndex(7,3)) == [FacetIndex(12,1)]
                    @test getneighborhood(topology, grid, FacetIndex(7,4)) == [FacetIndex(8,2)]
                    @test getneighborhood(topology, grid, FacetIndex(8,1)) == [FacetIndex(5,3)]
                    @test getneighborhood(topology, grid, FacetIndex(8,2)) == [FacetIndex(7,4)]
                    @test getneighborhood(topology, grid, FacetIndex(8,3)) == [FacetIndex(12,1)]
                    @test getneighborhood(topology, grid, FacetIndex(8,4)) == [FacetIndex(9,2)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(9,1)) == [FacetIndex(2,3)]
                    @test getneighborhood(topology, grid, FacetIndex(9,2)) == [FacetIndex(8,4), FacetIndex(5,4)]
                    @test getneighborhood(topology, grid, FacetIndex(9,3)) == [FacetIndex(12,1)]
                    @test getneighborhood(topology, grid, FacetIndex(9,4)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(10,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(10,2)) == [FacetIndex(11,4)]
                    @test getneighborhood(topology, grid, FacetIndex(10,3)) == [FacetIndex(13,1)]
                    @test getneighborhood(topology, grid, FacetIndex(10,4)) == [FacetIndex(3,2), FacetIndex(6,2), FacetIndex(7,2)]
                end
                @testset "coarsening" begin
                    cells_to_coarsen = Set([CellIndex(4)]);
                    coarsen!(grid, topology, refinement_cache, kopp_cache, kopp_values, cells_to_coarsen, dh);
                    @testset "coords" begin    
                        coords = zeros(Vec{2, Float64}, 4)
                        Ferrite.getcoordinates!(coords, grid, CellIndex(4))
                        @test coords ≈ [Vec(-4/6, -4/6), Vec(-2/6, -4/6), Vec(-2/6, -2/6), Vec(-4/6, -2/6)]
                        Ferrite.getcoordinates!(coords, grid, CellIndex(5))
                        @test coords ≈ [Vec(-1.0, -4/6), Vec(-4/6, -4/6), Vec(-4/6, -2/6), Vec(-1.0, -2/6)]
                    end
                    @testset "topology" begin
                        @test getneighborhood(topology, grid, FacetIndex(2,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(2,2)) == [FacetIndex(3,4)]
                        @test getneighborhood(topology, grid, FacetIndex(2,3)) == [FacetIndex(5,1)]
                        @test getneighborhood(topology, grid, FacetIndex(2,4)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(3,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(3,2)) == [FacetIndex(6,4)]
                        @test getneighborhood(topology, grid, FacetIndex(3,3)) == [FacetIndex(4,1)]
                        @test getneighborhood(topology, grid, FacetIndex(3,4)) == [FacetIndex(2,2)]
                        @test getneighborhood(topology, grid, FacetIndex(4,1)) == [FacetIndex(3,3)]
                        @test getneighborhood(topology, grid, FacetIndex(4,2)) == [FacetIndex(6,4)]
                        @test getneighborhood(topology, grid, FacetIndex(4,3)) == [FacetIndex(8,1)]
                        @test getneighborhood(topology, grid, FacetIndex(4,4)) == [FacetIndex(5,2)]
                        @test getneighborhood(topology, grid, FacetIndex(5,1)) == [FacetIndex(2,3)]
                        @test getneighborhood(topology, grid, FacetIndex(5,2)) == [FacetIndex(4,4)]
                        @test getneighborhood(topology, grid, FacetIndex(5,3)) == [FacetIndex(8,1)]
                        @test getneighborhood(topology, grid, FacetIndex(5,4)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(6,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(6,2)) == [FacetIndex(7,4)]
                        @test getneighborhood(topology, grid, FacetIndex(6,3)) == [FacetIndex(9,1)]
                        @test getneighborhood(topology, grid, FacetIndex(6,4)) == [FacetIndex(3,2), FacetIndex(4,2)]
                    end
                end
            end
        end
    end    
end

@testset "Kopp Hexahedron Refinement" begin
    ip = DiscontinuousLagrange{RefHexahedron, 1}()
    qr = QuadratureRule{RefHexahedron}(1);
    qr_facet = FacetQuadratureRule{RefHexahedron}(1);
    kopp_values = KoppValues(
        CellValues(qr, ip),
        FacetValues(qr_facet, ip),
        InterfaceValues(qr_facet, ip)
    );
    @testset "2x2 Hexahedron corner" begin
        grid = generate_grid(KoppCell{3, Int}, (2,2,2))
        topology = KoppTopology(grid)
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh);
        refinement_cache = KoppRefinementCache(topology);
        kopp_cache = KoppCache(grid, dh, kopp_values, refinement_cache, topology);
        cells_to_refine = Set([CellIndex(1)]);
        refine!(grid, topology, refinement_cache, kopp_cache, kopp_values, cells_to_refine, dh)
        @testset "one level" begin
            @testset "coords" begin    
                coords = zeros(Vec{3, Float64}, 8)
                Ferrite.getcoordinates!(coords, grid, CellIndex(2))
                @test coords ≈ [Vec(-1.0, -1.0, -1.0), Vec(-0.5, -1.0, -1.0),
                                Vec(-0.5, -0.5, -1.0), Vec(-1.0, -0.5, -1.0),
                                Vec(-1.0, -1.0, -0.5), Vec(-0.5, -1.0, -0.5),
                                Vec(-0.5, -0.5, -0.5), Vec(-1.0, -0.5, -0.5)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(3))
                @test coords ≈ [Vec(-0.5, -1.0, -1.0), Vec(0.0, -1.0, -1.0),
                                Vec(0.0, -0.5, -1.0), Vec(-0.5, -0.5, -1.0),
                                Vec(-0.5, -1.0, -0.5), Vec(0.0, -1.0, -0.5),
                                Vec(0.0, -0.5, -0.5), Vec(-0.5, -0.5, -0.5)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(4))
                @test coords ≈ [Vec(-0.5, -0.5, -1.0), Vec(0.0, -0.5, -1.0),
                                Vec(0.0, 0.0, -1.0), Vec(-0.5, 0.0, -1.0),
                                Vec(-0.5, -0.5, -0.5), Vec(0.0, -0.5, -0.5),
                                Vec(0.0, 0.0, -0.5), Vec(-0.5, 0.0, -0.5)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(5))
                @test coords ≈ [Vec(-1.0, -0.5, -1.0), Vec(-0.5, -0.5, -1.0),
                                Vec(-0.5, 0.0, -1.0), Vec(-1.0, 0.0, -1.0),
                                Vec(-1.0, -0.5, -0.5), Vec(-0.5, -0.5, -0.5),
                                Vec(-0.5, 0.0, -0.5), Vec(-1.0, 0.0, -0.5)]        
                Ferrite.getcoordinates!(coords, grid, CellIndex(6))
                @test coords ≈ [Vec(-1.0, -1.0, -0.5), Vec(-0.5, -1.0, -0.5),
                                Vec(-0.5, -0.5, -0.5), Vec(-1.0, -0.5, -0.5),
                                Vec(-1.0, -1.0, -0.0), Vec(-0.5, -1.0, -0.0),
                                Vec(-0.5, -0.5, -0.0), Vec(-1.0, -0.5, -0.0)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(7))
                @test coords ≈ [Vec(-0.5, -1.0, -0.5), Vec(0.0, -1.0, -0.5),
                                Vec(0.0, -0.5, -0.5), Vec(-0.5, -0.5, -0.5),
                                Vec(-0.5, -1.0, -0.0), Vec(0.0, -1.0, -0.0),
                                Vec(0.0, -0.5, -0.0), Vec(-0.5, -0.5, -0.0)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(8))
                @test coords ≈ [Vec(-0.5, -0.5, -0.5), Vec(0.0, -0.5, -0.5),
                                Vec(0.0, 0.0, -0.5), Vec(-0.5, 0.0, -0.5),
                                Vec(-0.5, -0.5, -0.0), Vec(0.0, -0.5, -0.0),
                                Vec(0.0, 0.0, -0.0), Vec(-0.5, 0.0, -0.0)]
                Ferrite.getcoordinates!(coords, grid, CellIndex(9))
                @test coords ≈ [Vec(-1.0, -0.5, -0.5), Vec(-0.5, -0.5, -0.5),
                                Vec(-0.5, 0.0, -0.5), Vec(-1.0, 0.0, -0.5),
                                Vec(-1.0, -0.5, -0.0), Vec(-0.5, -0.5, -0.0),
                                Vec(-0.5, 0.0, -0.0), Vec(-1.0, 0.0, -0.0)]        
            end
            @testset "topology" begin
                @test getneighborhood(topology, grid, FacetIndex(2,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(2,2)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(2,3)) == [FacetIndex(3,5)]
                @test getneighborhood(topology, grid, FacetIndex(2,4)) == [FacetIndex(5,2)]
                @test getneighborhood(topology, grid, FacetIndex(2,5)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(2,6)) == [FacetIndex(6,1)]

                @test getneighborhood(topology, grid, FacetIndex(3,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(3,2)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(3,3)) == [FacetIndex(10,5)]
                @test getneighborhood(topology, grid, FacetIndex(3,4)) == [FacetIndex(4,2)]
                @test getneighborhood(topology, grid, FacetIndex(3,5)) == [FacetIndex(2,3)]
                @test getneighborhood(topology, grid, FacetIndex(3,6)) == [FacetIndex(7,1)]

                @test getneighborhood(topology, grid, FacetIndex(4,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(4,2)) == [FacetIndex(3,4)]
                @test getneighborhood(topology, grid, FacetIndex(4,3)) == [FacetIndex(10,5)]
                @test getneighborhood(topology, grid, FacetIndex(4,4)) == [FacetIndex(11,2)]
                @test getneighborhood(topology, grid, FacetIndex(4,5)) == [FacetIndex(5,3)]
                @test getneighborhood(topology, grid, FacetIndex(4,6)) == [FacetIndex(8,1)]
                
                @test getneighborhood(topology, grid, FacetIndex(5,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(5,2)) == [FacetIndex(2,4)]
                @test getneighborhood(topology, grid, FacetIndex(5,3)) == [FacetIndex(4,5)]
                @test getneighborhood(topology, grid, FacetIndex(5,4)) == [FacetIndex(11,2)]
                @test getneighborhood(topology, grid, FacetIndex(5,5)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(5,6)) == [FacetIndex(9,1)]

                @test getneighborhood(topology, grid, FacetIndex(6,1)) == [FacetIndex(2,6)]
                @test getneighborhood(topology, grid, FacetIndex(6,2)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(6,3)) == [FacetIndex(7,5)]
                @test getneighborhood(topology, grid, FacetIndex(6,4)) == [FacetIndex(9,2)]
                @test getneighborhood(topology, grid, FacetIndex(6,5)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(6,6)) == [FacetIndex(13,1)]

                @test getneighborhood(topology, grid, FacetIndex(7,1)) == [FacetIndex(3,6)]
                @test getneighborhood(topology, grid, FacetIndex(7,2)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(7,3)) == [FacetIndex(10,5)]
                @test getneighborhood(topology, grid, FacetIndex(7,4)) == [FacetIndex(8,2)]
                @test getneighborhood(topology, grid, FacetIndex(7,5)) == [FacetIndex(6,3)]
                @test getneighborhood(topology, grid, FacetIndex(7,6)) == [FacetIndex(13,1)]

                @test getneighborhood(topology, grid, FacetIndex(8,1)) == [FacetIndex(4,6)]
                @test getneighborhood(topology, grid, FacetIndex(8,2)) == [FacetIndex(7,4)]
                @test getneighborhood(topology, grid, FacetIndex(8,3)) == [FacetIndex(10,5)]
                @test getneighborhood(topology, grid, FacetIndex(8,4)) == [FacetIndex(11,2)]
                @test getneighborhood(topology, grid, FacetIndex(8,5)) == [FacetIndex(9,3)]
                @test getneighborhood(topology, grid, FacetIndex(8,6)) == [FacetIndex(13,1)]
                
                @test getneighborhood(topology, grid, FacetIndex(9,1)) == [FacetIndex(5,6)]
                @test getneighborhood(topology, grid, FacetIndex(9,2)) == [FacetIndex(6,4)]
                @test getneighborhood(topology, grid, FacetIndex(9,3)) == [FacetIndex(8,5)]
                @test getneighborhood(topology, grid, FacetIndex(9,4)) == [FacetIndex(11,2)]
                @test getneighborhood(topology, grid, FacetIndex(9,5)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(9,6)) == [FacetIndex(13,1)]

                @test getneighborhood(topology, grid, FacetIndex(10,1)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(10,2)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(10,3)) === nothing
                @test getneighborhood(topology, grid, FacetIndex(10,4)) == [FacetIndex(12,2)]
                @test getneighborhood(topology, grid, FacetIndex(10,5)) == [FacetIndex(3,3), FacetIndex(4,3), FacetIndex(8,3), FacetIndex(7,3)]
                @test getneighborhood(topology, grid, FacetIndex(10,6)) == [FacetIndex(14,1)]
            end
            @testset "two levels" begin
                cells_to_refine = Set([CellIndex(8)]);
                refine!(grid, topology, refinement_cache, kopp_cache, kopp_values, cells_to_refine, dh)
                @testset "coords" begin
                    coords = zeros(Vec{3, Float64}, 8)
                    Ferrite.getcoordinates!(coords, grid, CellIndex(9))
                    @test coords ≈ [Vec(-0.5, -0.5, -0.5), Vec(-0.25, -0.5, -0.5),
                                    Vec(-0.25, -0.25, -0.5), Vec(-0.5, -0.25, -0.5),
                                    Vec(-0.5, -0.5, -0.25), Vec(-0.25, -0.5, -0.25),
                                    Vec(-0.25, -0.25, -0.25), Vec(-0.5, -0.25, -0.25)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(10))
                    @test coords ≈ [Vec(-0.25, -0.5, -0.5), Vec(0.0, -0.5, -0.5),
                                    Vec(0.0, -0.25, -0.5), Vec(-0.25, -0.25, -0.5),
                                    Vec(-0.25, -0.5, -0.25), Vec(0.0, -0.5, -0.25),
                                    Vec(0.0, -0.25, -0.25), Vec(-0.25, -0.25, -0.25)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(11))
                    @test coords ≈ [Vec(-0.25, -0.25, -0.5), Vec(0.0, -0.25, -0.5),
                                    Vec(0.0, 0.0, -0.5), Vec(-0.25, 0.0, -0.5),
                                    Vec(-0.25, -0.25, -0.25), Vec(0.0, -0.25, -0.25),
                                    Vec(0.0, 0.0, -0.25), Vec(-0.25, 0.0, -0.25)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(12))
                    @test coords ≈ [Vec(-0.5, -0.25, -0.5), Vec(-0.25, -0.25, -0.5),
                                    Vec(-0.25, 0.0, -0.5), Vec(-0.5, 0.0, -0.5),
                                    Vec(-0.5, -0.25, -0.25), Vec(-0.25, -0.25, -0.25),
                                    Vec(-0.25, 0.0, -0.25), Vec(-0.5, 0.0, -0.25)]        
                    Ferrite.getcoordinates!(coords, grid, CellIndex(13))
                    @test coords ≈ [Vec(-0.5, -0.5, -0.25), Vec(-0.25, -0.5, -0.25),
                                    Vec(-0.25, -0.25, -0.25), Vec(-0.5, -0.25, -0.25),
                                    Vec(-0.5, -0.5, -0.0), Vec(-0.25, -0.5, -0.0),
                                    Vec(-0.25, -0.25, -0.0), Vec(-0.5, -0.25, -0.0)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(14))
                    @test coords ≈ [Vec(-0.25, -0.5, -0.25), Vec(0.0, -0.5, -0.25),
                                    Vec(0.0, -0.25, -0.25), Vec(-0.25, -0.25, -0.25),
                                    Vec(-0.25, -0.5, -0.0), Vec(0.0, -0.5, -0.0),
                                    Vec(0.0, -0.25, -0.0), Vec(-0.25, -0.25, -0.0)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(15))
                    @test coords ≈ [Vec(-0.25, -0.25, -0.25), Vec(0.0, -0.25, -0.25),
                                    Vec(0.0, 0.0, -0.25), Vec(-0.25, 0.0, -0.25),
                                    Vec(-0.25, -0.25, -0.0), Vec(0.0, -0.25, -0.0),
                                    Vec(0.0, 0.0, -0.0), Vec(-0.25, 0.0, -0.0)]
                    Ferrite.getcoordinates!(coords, grid, CellIndex(16))
                    @test coords ≈ [Vec(-0.5, -0.25, -0.25), Vec(-0.25, -0.25, -0.25),
                                    Vec(-0.25, 0.0, -0.25), Vec(-0.5, 0.0, -0.25),
                                    Vec(-0.5, -0.25, -0.0), Vec(-0.25, -0.25, -0.0),
                                    Vec(-0.25, 0.0, -0.0), Vec(-0.5, 0.0, -0.0)]
                end
                @testset "topology" begin
                    @test getneighborhood(topology, grid, FacetIndex(2,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(2,2)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(2,3)) == [FacetIndex(3,5)]
                    @test getneighborhood(topology, grid, FacetIndex(2,4)) == [FacetIndex(5,2)]
                    @test getneighborhood(topology, grid, FacetIndex(2,5)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(2,6)) == [FacetIndex(6,1)]
    
                    @test getneighborhood(topology, grid, FacetIndex(3,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(3,2)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(3,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(3,4)) == [FacetIndex(4,2)]
                    @test getneighborhood(topology, grid, FacetIndex(3,5)) == [FacetIndex(2,3)]
                    @test getneighborhood(topology, grid, FacetIndex(3,6)) == [FacetIndex(7,1)]
    
                    @test getneighborhood(topology, grid, FacetIndex(4,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(4,2)) == [FacetIndex(3,4)]
                    @test getneighborhood(topology, grid, FacetIndex(4,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(4,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(4,5)) == [FacetIndex(5,3)]
                    @test getneighborhood(topology, grid, FacetIndex(4,6)) == [FacetIndex(9,1), FacetIndex(12,1), FacetIndex(11,1), FacetIndex(10,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(5,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(5,2)) == [FacetIndex(2,4)]
                    @test getneighborhood(topology, grid, FacetIndex(5,3)) == [FacetIndex(4,5)]
                    @test getneighborhood(topology, grid, FacetIndex(5,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(5,5)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(5,6)) == [FacetIndex(17,1)]
    
                    @test getneighborhood(topology, grid, FacetIndex(6,1)) == [FacetIndex(2,6)]
                    @test getneighborhood(topology, grid, FacetIndex(6,2)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(6,3)) == [FacetIndex(7,5)]
                    @test getneighborhood(topology, grid, FacetIndex(6,4)) == [FacetIndex(17,2)]
                    @test getneighborhood(topology, grid, FacetIndex(6,5)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(6,6)) == [FacetIndex(21,1)]
    
                    @test getneighborhood(topology, grid, FacetIndex(7,1)) == [FacetIndex(3,6)]
                    @test getneighborhood(topology, grid, FacetIndex(7,2)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(7,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(7,4)) == [FacetIndex(9,2), FacetIndex(10,2), FacetIndex(14,2), FacetIndex(13,2)]
                    @test getneighborhood(topology, grid, FacetIndex(7,5)) == [FacetIndex(6,3)]
                    @test getneighborhood(topology, grid, FacetIndex(7,6)) == [FacetIndex(21,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(9,1)) == [FacetIndex(4,6)]
                    @test getneighborhood(topology, grid, FacetIndex(9,2)) == [FacetIndex(7,4)]
                    @test getneighborhood(topology, grid, FacetIndex(9,3)) == [FacetIndex(10,5)]
                    @test getneighborhood(topology, grid, FacetIndex(9,4)) == [FacetIndex(12,2)]
                    @test getneighborhood(topology, grid, FacetIndex(9,5)) == [FacetIndex(17,3)]
                    @test getneighborhood(topology, grid, FacetIndex(9,6)) == [FacetIndex(13,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(10,1)) == [FacetIndex(4,6)]
                    @test getneighborhood(topology, grid, FacetIndex(10,2)) == [FacetIndex(7,4)]
                    @test getneighborhood(topology, grid, FacetIndex(10,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(10,4)) == [FacetIndex(11,2)]
                    @test getneighborhood(topology, grid, FacetIndex(10,5)) == [FacetIndex(9,3)]
                    @test getneighborhood(topology, grid, FacetIndex(10,6)) == [FacetIndex(14,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(11,1)) == [FacetIndex(4,6)]
                    @test getneighborhood(topology, grid, FacetIndex(11,2)) == [FacetIndex(10,4)]
                    @test getneighborhood(topology, grid, FacetIndex(11,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(11,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(11,5)) == [FacetIndex(12,3)]
                    @test getneighborhood(topology, grid, FacetIndex(11,6)) == [FacetIndex(15,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(12,1)) == [FacetIndex(4,6)]
                    @test getneighborhood(topology, grid, FacetIndex(12,2)) == [FacetIndex(9,4)]
                    @test getneighborhood(topology, grid, FacetIndex(12,3)) == [FacetIndex(11,5)]
                    @test getneighborhood(topology, grid, FacetIndex(12,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(12,5)) == [FacetIndex(17,3)]
                    @test getneighborhood(topology, grid, FacetIndex(12,6)) == [FacetIndex(16,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(13,1)) == [FacetIndex(9,6)]
                    @test getneighborhood(topology, grid, FacetIndex(13,2)) == [FacetIndex(7,4)]
                    @test getneighborhood(topology, grid, FacetIndex(13,3)) == [FacetIndex(14,5)]
                    @test getneighborhood(topology, grid, FacetIndex(13,4)) == [FacetIndex(16,2)]
                    @test getneighborhood(topology, grid, FacetIndex(13,5)) == [FacetIndex(17,3)]
                    @test getneighborhood(topology, grid, FacetIndex(13,6)) == [FacetIndex(21,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(14,1)) == [FacetIndex(10,6)]
                    @test getneighborhood(topology, grid, FacetIndex(14,2)) == [FacetIndex(7,4)]
                    @test getneighborhood(topology, grid, FacetIndex(14,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(14,4)) == [FacetIndex(15,2)]
                    @test getneighborhood(topology, grid, FacetIndex(14,5)) == [FacetIndex(13,3)]
                    @test getneighborhood(topology, grid, FacetIndex(14,6)) == [FacetIndex(21,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(15,1)) == [FacetIndex(11,6)]
                    @test getneighborhood(topology, grid, FacetIndex(15,2)) == [FacetIndex(14,4)]
                    @test getneighborhood(topology, grid, FacetIndex(15,3)) == [FacetIndex(18,5)]
                    @test getneighborhood(topology, grid, FacetIndex(15,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(15,5)) == [FacetIndex(16,3)]
                    @test getneighborhood(topology, grid, FacetIndex(15,6)) == [FacetIndex(21,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(16,1)) == [FacetIndex(12,6)]
                    @test getneighborhood(topology, grid, FacetIndex(16,2)) == [FacetIndex(13,4)]
                    @test getneighborhood(topology, grid, FacetIndex(16,3)) == [FacetIndex(15,5)]
                    @test getneighborhood(topology, grid, FacetIndex(16,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(16,5)) == [FacetIndex(17,3)]
                    @test getneighborhood(topology, grid, FacetIndex(16,6)) == [FacetIndex(21,1)]
                    
                    @test getneighborhood(topology, grid, FacetIndex(17,1)) == [FacetIndex(5,6)]
                    @test getneighborhood(topology, grid, FacetIndex(17,2)) == [FacetIndex(6,4)]
                    @test getneighborhood(topology, grid, FacetIndex(17,3)) == [FacetIndex(9,5), FacetIndex(13,5), FacetIndex(16,5), FacetIndex(12,5)]
                    @test getneighborhood(topology, grid, FacetIndex(17,4)) == [FacetIndex(19,2)]
                    @test getneighborhood(topology, grid, FacetIndex(17,5)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(17,6)) == [FacetIndex(21,1)]
    
                    @test getneighborhood(topology, grid, FacetIndex(18,1)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(18,2)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(18,3)) === nothing
                    @test getneighborhood(topology, grid, FacetIndex(18,4)) == [FacetIndex(20,2)]
                    @test getneighborhood(topology, grid, FacetIndex(18,5)) == [FacetIndex(3,3), FacetIndex(4,3), FacetIndex(10,3), FacetIndex(11,3), FacetIndex(15,3), FacetIndex(14,3), FacetIndex(7,3)]
                    @test getneighborhood(topology, grid, FacetIndex(18,6)) == [FacetIndex(22,1)]
                end
                @testset "coarsening" begin
                    cells_to_coarsen = Set([CellIndex(8)])
                    coarsen!(grid, topology, refinement_cache, kopp_cache, kopp_values, cells_to_coarsen, dh);
                    @testset "coords" begin    
                        coords = zeros(Vec{3, Float64}, 8)
                        Ferrite.getcoordinates!(coords, grid, CellIndex(8))
                        @test coords ≈ [Vec(-0.5, -0.5, -0.5), Vec(0.0, -0.5, -0.5),
                                        Vec(0.0, 0.0, -0.5), Vec(-0.5, 0.0, -0.5),
                                        Vec(-0.5, -0.5, -0.0), Vec(0.0, -0.5, -0.0),
                                        Vec(0.0, 0.0, -0.0), Vec(-0.5, 0.0, -0.0)]
                        Ferrite.getcoordinates!(coords, grid, CellIndex(9))
                        @test coords ≈ [Vec(-1.0, -0.5, -0.5), Vec(-0.5, -0.5, -0.5),
                                        Vec(-0.5, 0.0, -0.5), Vec(-1.0, 0.0, -0.5),
                                        Vec(-1.0, -0.5, -0.0), Vec(-0.5, -0.5, -0.0),
                                        Vec(-0.5, 0.0, -0.0), Vec(-1.0, 0.0, -0.0)]        
                    end
                    @testset "topology" begin
                        @test getneighborhood(topology, grid, FacetIndex(2,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(2,2)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(2,3)) == [FacetIndex(3,5)]
                        @test getneighborhood(topology, grid, FacetIndex(2,4)) == [FacetIndex(5,2)]
                        @test getneighborhood(topology, grid, FacetIndex(2,5)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(2,6)) == [FacetIndex(6,1)]
        
                        @test getneighborhood(topology, grid, FacetIndex(3,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(3,2)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(3,3)) == [FacetIndex(10,5)]
                        @test getneighborhood(topology, grid, FacetIndex(3,4)) == [FacetIndex(4,2)]
                        @test getneighborhood(topology, grid, FacetIndex(3,5)) == [FacetIndex(2,3)]
                        @test getneighborhood(topology, grid, FacetIndex(3,6)) == [FacetIndex(7,1)]
        
                        @test getneighborhood(topology, grid, FacetIndex(4,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(4,2)) == [FacetIndex(3,4)]
                        @test getneighborhood(topology, grid, FacetIndex(4,3)) == [FacetIndex(10,5)]
                        @test getneighborhood(topology, grid, FacetIndex(4,4)) == [FacetIndex(11,2)]
                        @test getneighborhood(topology, grid, FacetIndex(4,5)) == [FacetIndex(5,3)]
                        @test getneighborhood(topology, grid, FacetIndex(4,6)) == [FacetIndex(8,1)]
                        
                        @test getneighborhood(topology, grid, FacetIndex(5,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(5,2)) == [FacetIndex(2,4)]
                        @test getneighborhood(topology, grid, FacetIndex(5,3)) == [FacetIndex(4,5)]
                        @test getneighborhood(topology, grid, FacetIndex(5,4)) == [FacetIndex(11,2)]
                        @test getneighborhood(topology, grid, FacetIndex(5,5)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(5,6)) == [FacetIndex(9,1)]
        
                        @test getneighborhood(topology, grid, FacetIndex(6,1)) == [FacetIndex(2,6)]
                        @test getneighborhood(topology, grid, FacetIndex(6,2)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(6,3)) == [FacetIndex(7,5)]
                        @test getneighborhood(topology, grid, FacetIndex(6,4)) == [FacetIndex(9,2)]
                        @test getneighborhood(topology, grid, FacetIndex(6,5)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(6,6)) == [FacetIndex(13,1)]
        
                        @test getneighborhood(topology, grid, FacetIndex(7,1)) == [FacetIndex(3,6)]
                        @test getneighborhood(topology, grid, FacetIndex(7,2)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(7,3)) == [FacetIndex(10,5)]
                        @test getneighborhood(topology, grid, FacetIndex(7,4)) == [FacetIndex(8,2)]
                        @test getneighborhood(topology, grid, FacetIndex(7,5)) == [FacetIndex(6,3)]
                        @test getneighborhood(topology, grid, FacetIndex(7,6)) == [FacetIndex(13,1)]
        
                        @test getneighborhood(topology, grid, FacetIndex(8,1)) == [FacetIndex(4,6)]
                        @test getneighborhood(topology, grid, FacetIndex(8,2)) == [FacetIndex(7,4)]
                        @test getneighborhood(topology, grid, FacetIndex(8,3)) == [FacetIndex(10,5)]
                        @test getneighborhood(topology, grid, FacetIndex(8,4)) == [FacetIndex(11,2)]
                        @test getneighborhood(topology, grid, FacetIndex(8,5)) == [FacetIndex(9,3)]
                        @test getneighborhood(topology, grid, FacetIndex(8,6)) == [FacetIndex(13,1)]
                        
                        @test getneighborhood(topology, grid, FacetIndex(9,1)) == [FacetIndex(5,6)]
                        @test getneighborhood(topology, grid, FacetIndex(9,2)) == [FacetIndex(6,4)]
                        @test getneighborhood(topology, grid, FacetIndex(9,3)) == [FacetIndex(8,5)]
                        @test getneighborhood(topology, grid, FacetIndex(9,4)) == [FacetIndex(11,2)]
                        @test getneighborhood(topology, grid, FacetIndex(9,5)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(9,6)) == [FacetIndex(13,1)]
        
                        @test getneighborhood(topology, grid, FacetIndex(10,1)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(10,2)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(10,3)) === nothing
                        @test getneighborhood(topology, grid, FacetIndex(10,4)) == [FacetIndex(12,2)]
                        @test getneighborhood(topology, grid, FacetIndex(10,5)) == [FacetIndex(8,3), FacetIndex(3,3), FacetIndex(7,3), FacetIndex(4,3)]
                        @test getneighborhood(topology, grid, FacetIndex(10,6)) == [FacetIndex(14,1)]
                    end
                end
            end
        end
    end
end
