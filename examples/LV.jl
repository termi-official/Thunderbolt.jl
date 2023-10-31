include("common-stuff.jl")
using FerriteGmsh

# TODO refactor this one into the framework code and put a nice abstraction layer around it
struct StandardMechanicalIOPostProcessor2{IO, CV, CC, MC}
    io::IO
    cv::CV
    subdomains::Vector{Int}
    coordinate_system::CC
    microstructure_cache::MC
end

function (postproc::StandardMechanicalIOPostProcessor2{IO, CV, MC})(t, problem, solver_cache) where {IO, CV, MC}
    @unpack dh = problem
    grid = Ferrite.get_grid(dh)
    @unpack io, cv, subdomains, coordinate_system, microstructure_cache = postproc

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
    helixangledata = zero(Vector{Float64}(undef, getncells(grid)))
    helixanglerefdata = zero(Vector{Float64}(undef, getncells(grid)))

    # Compute some elementwise measures
    for sdh ∈ dh.subdofhandlers[postproc.subdomains]
        field_idx = Ferrite.find_field(sdh, :displacement)
        field_idx === nothing && continue 
        for cell ∈ CellIterator(sdh)
            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            field_dofs  = dof_range(sdh, field_idx)
            uₑ = solver_cache.uₜ[global_dofs] # element dofs

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
            helixangle_cell = 0.0
            helixangleref_cell = 0.0

            nqp = getnquadpoints(cv)
            for qpᵢ in 1:nqp
                qp = QuadraturePoint(qpᵢ, cv.qr.points[qpᵢ])
                dΩ = getdetJdV(cv, qpᵢ)

                # Compute deformation gradient F
                ∇u = function_gradient(cv, qpᵢ, uₑ)
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
                x_global = spatial_coordinate(cv, qpᵢ, coords)

                # v_longitudinal = function_gradient(cv_cs, qp, coordinate_system.u_apicobasal[celldofs(cell)])
                # v_radial = function_gradient(cv_cs, qp, coordinate_system.u_transmural[celldofs(cell)])
                # v_circimferential = v_longitudinal × v_radial
                # @TODO compute properly via coordinate system
                v_longitudinal = Ferrite.Vec{3}((0.0, 0.0, 1.0))
                v_radial = Ferrite.Vec{3}((x_global[1],x_global[2],0.0))
                v_radial /= norm(v_radial)
                v_circimferential = v_longitudinal × v_radial # Ferrite.Vec{3}((x_global[2],x_global[1],0.0))
                v_circimferential /= norm(v_circimferential)
                #
                E_ll_cell += v_longitudinal ⋅ E ⋅ v_longitudinal
                E_rr_cell += v_radial ⋅ E ⋅ v_radial
                E_cc_cell += v_circimferential ⋅ E ⋅ v_circimferential

                Jdata_cell += det(F)

                frefdata_cell += f₀
                srefdata_cell += s₀

                fdata_cell += f₀_current
                sdata_cell += s₀_current

                helixangle_cell += acos(clamp(f₀_current ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀_current) ⋅ v_radial)
                helixangleref_cell += acos(clamp(f₀ ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀) ⋅ v_radial)
            end

            E_ff[Ferrite.cellid(cell)] = E_ff_cell / nqp
            E_cc[Ferrite.cellid(cell)] = E_cc_cell / nqp
            E_rr[Ferrite.cellid(cell)] = E_rr_cell / nqp
            E_ll[Ferrite.cellid(cell)] = E_ll_cell / nqp
            Jdata[Ferrite.cellid(cell)] = Jdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] = frefdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] /= norm(frefdata[Ferrite.cellid(cell)])
            srefdata[Ferrite.cellid(cell)] = srefdata_cell / nqp
            srefdata[Ferrite.cellid(cell)] /= norm(srefdata[Ferrite.cellid(cell)])
            fdata[Ferrite.cellid(cell)] = fdata_cell / nqp
            fdata[Ferrite.cellid(cell)] /= norm(fdata[Ferrite.cellid(cell)])
            sdata[Ferrite.cellid(cell)] = sdata_cell / nqp
            sdata[Ferrite.cellid(cell)] /= norm(sdata[Ferrite.cellid(cell)])
            helixanglerefdata[Ferrite.cellid(cell)] = helixangleref_cell / nqp
            helixangledata[Ferrite.cellid(cell)] = helixangle_cell / nqp
        end
    end

    # Save the solution
    Thunderbolt.store_timestep!(io, t, dh.grid)
    Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₜ, :displacement)
    Thunderbolt.store_timestep_field!(io, t, coordinate_system.dh, coordinate_system.u_transmural, "transmural")
    Thunderbolt.store_timestep_field!(io, t, coordinate_system.dh, coordinate_system.u_apicobasal, "apicobasal")
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
    Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixangledata),"Helix Angle")
    Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixanglerefdata),"Helix Angle (End Diastole)")
    Thunderbolt.finalize_timestep!(io, t)

    @show Thunderbolt.compute_chamber_volume(dh, solver_cache.uₜ, "Endocardium", Thunderbolt.Hirschvogel2016SurrogateVolume())
    # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
    # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
end


function solve_ideal_lv(name_base, material_model, grid, coordinate_system, microstructure_model, face_models::FM, calcium_field, ip_mech::IPM, ip_geo::IPG, intorder::Int, Δt = 0.1, T = 2.0) where {ref_shape, IPM <: Interpolation{ref_shape}, IPG <: Interpolation{ref_shape}, FM}
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    problem = semidiscretize(
        SimpleChamberContractionModel(material_model, calcium_field, face_models, microstructure_model),
        FiniteElementDiscretization(
            Dict(:displacement => LagrangeCollection{1}()^3),
            Dirichlet[],
            # [Dirichlet(:displacement, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])],
            # [Dirichlet(:displacement, getfaceset(grid, "left"), (x,t) -> [0.0], [1]),Dirichlet(:displacement, getfaceset(grid, "front"), (x,t) -> [0.0], [2]),Dirichlet(:displacement, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3]), Dirichlet(:displacement, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])],
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValues(QuadratureRule{ref_shape}(intorder-1), ip_mech, ip_geo)
    microstructure_cache = setup_microstructure_cache(cv_post, microstructure_model, CellCache(problem.dh)) # HOTFIX CTOR
    standard_postproc = StandardMechanicalIOPostProcessor2(io, cv_post, [1], coordinate_system, microstructure_cache)

    # Create sparse matrix and residual vector
    solver = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100))

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        nothing,
        standard_postproc
    )
end

# LV_grid = Thunderbolt.hexahedralize(Thunderbolt.generate_ideal_lv_mesh(15,2,6))
# ref_shape = RefHexahedron
LV_grid = togrid("../data/meshes/LV/EllipsoidalLeftVentricle.msh")
ref_shape = RefTetrahedron
order = 1
ip = Lagrange{ref_shape, order}()^3
ip_fiber = Lagrange{ref_shape, order}()
ip_geo = Lagrange{ref_shape, order}()
LV_cs = compute_LV_coordinate_system(LV_grid, ip)
LV_fm = create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = -60.0, epi_helix_angle = 70.0, endo_transversal_angle = 10.0, epi_transversal_angle = -20.0)
passive_ho_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))
solve_ideal_lv("LV_test",
    ActiveStressModel(
        passive_ho_model,
        SimpleActiveStress(10.0),
        PelceSunLangeveld1995Model()
    ), LV_grid, LV_cs, LV_fm,
    [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
    ip, ip_geo, max(2*order-1,2)
)

# LV_grid = togrid("data/meshes/LV/EllipsoidalLeftVentricleQuadTet.msh")
# ref_shape = RefTetrahedron
# order = 2
# ip_fiber = Lagrange{ref_shape, order}()
# ip     = Lagrange{ref_shape, order}()^3
# ip_geo = Lagrange{ref_shape, order}()

# LV_cs = compute_LV_coordinate_system(LV_grid, ip)
# LV_fm = create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = -60.0, epi_helix_angle = 70.0, endo_transversal_angle = 10.0, epi_transversal_angle = -20.0)

# passive_ho_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))

# solve_ideal_lv("LV3_GHM_BNH_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         Thunderbolt.BioNeoHooekean(),
#         ActiveMaterialAdapter(NewActiveSpring(; aᶠ=5.0)),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_GHM-HO_AS1_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_GHM-HO_AS2_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring2()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_GHM-HO_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_GHM-HO_HO_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(passive_ho_model),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_ActiveStress-HO_Simple_Pelce",
#     ActiveStressModel(
#         passive_ho_model,
#         SimpleActiveStress(),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_ActiveStress-HO_Piersanti_Pelce",
#     ActiveStressModel(
#         passive_ho_model,
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# LV_grid = togrid("../data/meshes/LV/EllipsoidalLeftVentricle.msh")
# ref_shape = RefTetrahedron
# order = 1
# ip_fiber = Lagrange{ref_shape, order}()
# ip     = Lagrange{ref_shape, order}()^3
# ip_geo = Lagrange{ref_shape, order}()

# LV_cs = compute_LV_coordinate_system(LV_grid, ip)
# LV_fm = create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = -60.0, epi_helix_angle = 70.0, endo_transversal_angle = 10.0, epi_transversal_angle = -20.0)

# passive_ho_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(20.0))

# solve_ideal_lv("LV_ActiveStress-HO_Piersanti_Pelce-BCdriven",
#     ActiveStressModel(
#         passive_ho_model,
#         PiersantiActiveStress(10.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(10.0, "Epicardium"), NormalSpringBC(5.0, "Base")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV2_GHM_BNH_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         Thunderbolt.BioNeoHooekean(),
#         ActiveMaterialAdapter(NewActiveSpring(; aᶠ=5.0)),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV2_ActiveStress-HO_Piersanti_Pelce",
#     ActiveStressModel(
#         passive_ho_model,
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# ref_shape = RefTetrahedron
# order = 2
# ip_fiber = Lagrange{ref_shape, order}()
# ip     = Lagrange{ref_shape, order}()^3
# ip_geo = Lagrange{ref_shape, order}()

# LV_grid = togrid("../data/meshes/LV/EllipsoidalLeftVentricleQuadTet.msh")
# LV_cs = compute_LV_coordinate_system(LV_grid, ip_geo)
# LV_fm = create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(60.0), epi_helix_angle = deg2rad(-60.0), endo_transversal_angle = deg2rad(20.0), epi_transversal_angle = deg2rad(20.0))

# passive_honi_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, NeffCompressionPenalty(;β=10.0))

# solve_ideal_lv("LV3_GHM-HONI_AS1_RLRSQ75_Pelce-BCdriven",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring(;aᶠ=20.0)),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model(;λᵃₘₐₓ=0.6)
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(10.0, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_GHM-HONI_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_honi_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV3_GHM-HO2NI_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         HolzapfelOgden2009Model(;mpU=NeffCompressionPenalty(;β=5.0)),
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )


# solve_ideal_lv("BarbarottaRossiDedeQuarteroni2018-Reproducer",
#     GeneralizedHillModel(
#         NullEnergyModel(),
#         ActiveMaterialAdapter(HolzapfelOgden2009Model(;a=0.2, b=4.61, aᶠ=4.19, bᶠ=7.85, aˢ=2.56, bˢ=10.44, aᶠˢ=0.13, bᶠˢ=15.25, mpU=SimpleCompressionPenalty())),
#         RLRSQActiveDeformationGradientModel(0.25),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [RobinBC(7.5, "Epicardium"), NormalSpringBC(10.0, "Base"), ConstantPressureBC(0.14, "Endocardium")],
#     CalciumHatField(), ip, ip_geo, max(2*order-1,2)
# )


# solve_ideal_lv("BarbarottaRossiDedeQuarteroni2018-Adjusted",
#     GeneralizedHillModel(
#         NullEnergyModel(),
#         ActiveMaterialAdapter(HolzapfelOgden2009Model(;a=0.2, b=4.61, aᶠ=4.19, bᶠ=7.85, aˢ=2.56, bˢ=10.44, aᶠˢ=0.13, bᶠˢ=15.25, mpU=SimpleCompressionPenalty())),
#         RLRSQActiveDeformationGradientModel(0.25),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [RobinBC(0.01, "Epicardium"), NormalSpringBC(1.0, "Base"), ConstantPressureBC(0.0, "Endocardium")],
#     CalciumHatField(), ip, ip_geo, max(2*order-1,2)
# )

# FINALLY SOME SHEETLET REORIENTATION!
# LV_fm = create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(60.0), epi_helix_angle = deg2rad(-60.0), endo_transversal_angle = deg2rad(20.0), epi_transversal_angle = deg2rad(20.0), sheetlet_pseudo_angle=deg2rad(20.0), make_orthogonal=false)
# solve_ideal_lv("Vallespin2023-Reproducer",
#     ActiveStressModel(
#         Guccione1991Passive(),
#         Guccione1993Active(),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, LV_fm,
#     [NormalSpringBC(10.0, "Epicardium"), NormalSpringBC(10.0, "Base")],
#     CalciumHatField(), ip, ip_geo, max(2*order-1,2),
#     0.025
# )

# LV_grid = togrid("../data/meshes/LV/EllipsoidalLeftVentricle_fine.msh")
# ref_shape = RefTetrahedron
# order = 1
# ip = ip_geo = Lagrange{ref_shape, order}()

# passive_ho_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(10.0))

# LV_cs = compute_LV_coordinate_system(LV_grid, ip)

# solve_ideal_lv("LV-fine_GHM_HO_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(-60.0), epi_helix_angle = deg2rad(60.0)),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV-fine_GHM_HO_AS1_RLRSQ75_Pelce_SA20",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(-60.0), epi_helix_angle = deg2rad(60.0), sheetlet_pseudo_angle = deg2rad(20.0)),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )

# solve_ideal_lv("LV-fine_GHM_HO_AS1_RLRSQ75_Pelce_SA20_TA20",
#     GeneralizedHillModel(
#         passive_ho_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), LV_grid, LV_cs, create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(-60.0), epi_helix_angle = deg2rad(60.0), endo_transversal_angle = deg2rad(20.0), epi_transversal_angle = deg2rad(20.0), sheetlet_pseudo_angle = deg2rad(20.0)),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip, ip_geo, max(2*order-1,2)
# )
