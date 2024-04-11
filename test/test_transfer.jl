@testset "Transfer Opeartors" begin
    source_mesh = Thunderbolt.generate_simple_disc_mesh(Quadrilateral, 40)

    target_mesh = generate_mesh(Triangle, (10, 11))
    cells_hole = Set{Int}()
    cells_remaining = Set{Int}()
    for cc in CellIterator(target_mesh.grid)
        if all(norm.(getcoordinates(cc)) .≤ 1)
            push!(cells_hole, cellid(cc))
        else
            push!(cells_remaining, cellid(cc))
        end
    end

    @testset "Nodal Intergrid Interpolation" begin
        source_dh = DofHandler(source_mesh)
        add!(source_dh, :u, Lagrange{RefQuadrilateral,2}())
        add!(source_dh, :v, Lagrange{RefQuadrilateral,3}())
        close!(source_dh)

        source_u = ones(ndofs(source_dh))
        apply_analytical!(source_u, source_dh, :v, x->-norm(x))

        target_dh = DofHandler(target_mesh)
        target_sdh_hole = SubDofHandler(target_dh, cells_hole)
        add!(target_sdh_hole, :v, Lagrange{RefTriangle,2}())
        add!(target_sdh_hole, :w, Lagrange{RefTriangle,1}())
        close!(target_dh)

        target_u = [NaN for i in 1:ndofs(target_dh)]

        op = Thunderbolt.NodalIntergridInterpolation(source_dh, target_dh, :v)
        Thunderbolt.transfer!(target_u, op, source_u)
        cv = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle,2}())
        v_range = dof_range(target_dh.subdofhandlers[1], :v)
        w_range = dof_range(target_dh.subdofhandlers[1], :w)
        for cc in CellIterator(target_dh.subdofhandlers[1])
            reinit!(cv, cc)
            dofs_v = celldofs(cc)[v_range]
            for qp in QuadratureIterator(cv)
                x = Thunderbolt._spatial_coordinate(Lagrange{RefTriangle,1}(), qp.ξ, getcoordinates(cc))
                @test function_value(cv, qp, target_u[dofs_v]) ≈ -norm(x) atol=3e-1
            end
        end
    end
end
