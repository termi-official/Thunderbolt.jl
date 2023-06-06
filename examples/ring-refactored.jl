include("common-stuff.jl")
using FerriteGmsh


function solve_test_ring(pv_name_base, material_model, grid, coordinate_system, microstructure_model, face_models, calcium_field, ip_mech::Interpolation{3, ref_shape}, ip_geo::Interpolation{3, ref_shape}, intorder, Δt = 0.1) where {ref_shape}
    io = ParaViewWriter(pv_name_base);

    T = 2.0

    # Finite element base
    qr = QuadratureRule{3, ref_shape}(intorder)
    qr_face = QuadratureRule{2, ref_shape}(intorder)
    cv = CellVectorValues(qr, ip_mech, ip_geo)
    fv = FaceVectorValues(qr_face, ip_mech, ip_geo)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    dbcs = ConstraintHandler(dh)
    # Clamp three sides
    dbc = Dirichlet(:u, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])
    add!(dbcs, dbc)
    # dbc = Dirichlet(:u, Set([first(getfaceset(grid, "Base"))]), (x,t) -> [0.0], [3])
    # add!(dbcs, dbc)
    # dbc = Dirichlet(:u, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    # add!(dbcs, dbc)

    close!(dbcs)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)

    uₜ   = zeros(_ndofs)
    uₜ₋₁ = zeros(_ndofs)
    Δu   = zeros(_ndofs)

    # ref_vol = calculate_volume_deformed_mesh(uₜ,dh,cv);
    # min_vol = ref_vol
    # max_vol = ref_vol

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    f = zeros(_ndofs)

    NEWTON_TOL = 1e-8
    MAX_NEWTON_ITER = 100

    for t ∈ 0.0:Δt:T
        # Store last solution
        uₜ₋₁ .= uₜ

        # Update with new boundary conditions (if available)
        Ferrite.update!(dbcs, t)
        apply!(uₜ, dbcs)

        # Perform Newton iterations
        newton_itr = -1
        while true
            newton_itr += 1

            assemble_global!(K, f, dh, cv, fv, material_model, uₜ, uₜ₋₁, microstructure_model, calcium_field, t, Δt, face_models)

            rhsnorm = norm(f[Ferrite.free_dofs(dbcs)])
            apply_zero!(K, f, dbcs)
            @info t rhsnorm

            if rhsnorm < NEWTON_TOL
                break
            elseif newton_itr > MAX_NEWTON_ITER
                error("Reached maximum Newton iterations. Aborting.")
            end

            #@info det(K)
            try
                Δu = K \ f
            catch
                finalize!(io)
                @warn "Failed Solve at " t
                return uₜ
            end

            apply_zero!(Δu, dbcs)

            uₜ .-= Δu # Current guess
        end

        # Compute some elementwise measures
        E_ff = zeros(getncells(grid))
        E_ff2 = zeros(getncells(grid))
        E_cc = zeros(getncells(grid))
        E_ll = zeros(getncells(grid))
        E_rr = zeros(getncells(grid))

        Jdata = zeros(getncells(grid))

        frefdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
        srefdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
        fdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
        sdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))

        microstructure_cache = setup_microstructure_cache(cv, microstructure_model)

        for cell in CellIterator(dh)
            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            uₑ = uₜ[global_dofs] # element dofs

            update_microstructure_cache!(microstructure_cache, t, cell, cv)
                
            E_ff_cell = 0.0
            E_cc_cell = 0.0
            E_rr_cell = 0.0
            E_ll_cell = 0.0

            Jdata_cell = 0.0
            frefdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            srefdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            fdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            sdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))

            nqp = getnquadpoints(cv)
            for qp in 1:nqp
                dΩ = getdetJdV(cv, qp)

                # Compute deformation gradient F
                ∇u = function_gradient(cv, qp, uₑ)
                F = one(∇u) + ∇u

                C = tdot(F)
                E = (C-one(C))/2.0
                f₀,s₀,n₀ = directions(microstructure_cache, qp)

                E_ff_cell += f₀ ⋅ E ⋅ f₀

                f₀_current = F⋅f₀
                f₀_current /= norm(f₀_current)

                s₀_current = F⋅s₀
                s₀_current /= norm(s₀_current)

                coords = getcoordinates(cell)
                x_global = spatial_coordinate(cv, qp, coords)
                # @TODO compute properly
                v_longitudinal = Ferrite.Vec{3}((0.0, 0.0, 1.0))
                v_radial = Ferrite.Vec{3}((x_global[1],x_global[2],0.0))/norm(Ferrite.Vec{3}((x_global[1],x_global[2],0.0)))
                v_circimferential = Ferrite.Vec{3}((x_global[2],-x_global[1],0.0))/norm(Ferrite.Vec{3}((x_global[2],-x_global[1],0.0)))
                #
                E_ll_cell += v_longitudinal ⋅ E ⋅ v_longitudinal
                E_rr_cell += v_radial ⋅ E ⋅ v_radial
                E_cc_cell += v_circimferential ⋅ E ⋅ v_circimferential

                Jdata_cell += det(F)

                frefdata_cell += f₀
                srefdata_cell += s₀

                fdata_cell += f₀_current
                sdata_cell += s₀_current
            end

            E_ff[Ferrite.cellid(cell)] = E_ff_cell / nqp
            E_cc[Ferrite.cellid(cell)] = E_cc_cell / nqp
            E_rr[Ferrite.cellid(cell)] = E_rr_cell / nqp
            E_ll[Ferrite.cellid(cell)] = E_ll_cell / nqp
            Jdata[Ferrite.cellid(cell)] = Jdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] = frefdata_cell / nqp
            srefdata[Ferrite.cellid(cell)] = srefdata_cell / nqp
            fdata[Ferrite.cellid(cell)] = fdata_cell / nqp
            sdata[Ferrite.cellid(cell)] = sdata_cell / nqp
        end

        # Save the solution
        Thunderbolt.store_timestep!(io, t, dh, uₜ)
        Thunderbolt.store_timestep_celldata!(io, t, hcat(frefdata...),"Reference Fiber Data")
        Thunderbolt.store_timestep_celldata!(io, t, hcat(fdata...),"Current Fiber Data")
        Thunderbolt.store_timestep_celldata!(io, t, hcat(srefdata...),"Reference Sheet Data")
        Thunderbolt.store_timestep_celldata!(io, t, hcat(sdata...),"Current Sheet Data")
        Thunderbolt.store_timestep_celldata!(io, t, E_ff,"E_ff")
        Thunderbolt.store_timestep_celldata!(io, t, E_ff2,"E_ff2")
        Thunderbolt.store_timestep_celldata!(io, t, E_cc,"E_cc")
        Thunderbolt.store_timestep_celldata!(io, t, E_rr,"E_rr")
        Thunderbolt.store_timestep_celldata!(io, t, E_ll,"E_ll")
        Thunderbolt.store_timestep_celldata!(io, t, Jdata,"J")
        Thunderbolt.finalize_timestep!(io, t)

        # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
        # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
    end

    # println("Compression: ", (ref_vol/min_vol - 1.0)*100, "%")
    # println("Expansion: ", (ref_vol/max_vol - 1.0)*100, "%")

    finalize!(io)

    return uₜ
end

for (filename, ref_shape, order) ∈ [
    # ("MidVentricularSectionQuadTet.msh", RefTetrahedron, 2),
    ("MidVentricularSectionTet.msh", RefTetrahedron, 1),
    # ("MidVentricularSectionHex.msh", RefCube, 1),
    # ("MidVentricularSectionQuadHex.msh", RefCube, 2) # We have to update FerriteGmsh first, because the hex27 translator is missing. See https://github.com/Ferrite-FEM/FerriteGmsh.jl/pull/29
]

ip_fiber = Lagrange{3, ref_shape, order}()
ip_geo = Lagrange{3, ref_shape, order}()

ring_grid = saved_file_to_grid("../data/meshes/ring/" * filename)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
ring_fm = create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_angle = 60.0, epi_angle = -70.0, endo_transversal_angle = -10.0, epi_transversal_angle = 20.0)

# passive_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))

# solve_test_ring(filename*"_GHM-HO_AS1_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ), 
#     ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# Diverges...?
# solve_test_ring(filename*"_GHM-HO_AS2_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring2()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_HO_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(passive_model),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Simple_Pelce",
#     ActiveStressModel(
#         passive_model,
#         SimpleActiveStress(),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Piersanti_Pelce",
#     ActiveStressModel(
#         passive_model,
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

solve_test_ring(filename*"_GHM-HONEW_HONEW_RLRSQ100_Pelce",
    ExtendedHillModel(
        NewHolzapfelOgden2009Model(),
        ActiveMaterialAdapter(NewHolzapfelOgden2009Model(;mpU=NullCompressionPenalty())),
        RLRSQActiveDeformationGradientModel(0.5),
        PelceSunLangeveld1995Model()
    ), ring_grid, ring_cs, ring_fm, 
    [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
    ip_geo, ip_geo, 2*order
)

solve_test_ring(filename*"_ActiveStress-HONEW_Piersanti_Pelce",
    ActiveStressModel(
        NewHolzapfelOgden2009Model(),
        PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
        PelceSunLangeveld1995Model()
    ), ring_grid, ring_cs, ring_fm, 
    [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
    ip_geo, ip_geo, 2*order
)

end
