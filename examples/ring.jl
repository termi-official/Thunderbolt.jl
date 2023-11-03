include("common-stuff.jl")
using FerriteGmsh, DelimitedFiles

import Ferrite: get_grid, find_field

# mutable struct AverageCoordinateStrainProcessorCache{CC, SVT <: AbstractVector}
#     coordinate_system_cache::CC
#     E_cc::SVT
#     E_ll::SVT
#     E_rr::SVT
# end

# function prepare_postprocessor_cache!(cache::AverageCoordinateStrainProcessorCache{CC, SVT}, problem) where {CC, SVT <: AbstractVector}
#     @unpack dh   = problem
#     @unpack grid = dh
#     _ncells = getncells(grid)
#     cache.E_cc = zero(SVT(undef, _ncells))
#     cache.E_ll = zero(SVT(undef, _ncells))
#     cache.E_rr = zero(SVT(undef, _ncells))
# end

# mutable struct AverageCardiacMechanicalMicrostructuralPostProcessorCache{MC, SVT <: AbstractVector, VVT <: AbstractVector}
#     microstructure_cache::MC
#     E_ff:SVT
#     frefdata::VVT
#     fdata::VVT
#     srefdata::VVT
#     sdata::VVT
#     nrefdata::VVT
#     ndata::VVT
#     helixangledata::SVT
#     helixanglerefdata::SVT
# end

# function prepare_postprocessor_cache!(cache::AverageCardiacMechanicalMicrostructuralPostProcessorCache{MC, SVT, VVT}, problem, t, uₙ) where {MC, SVT <: AbstractVector, VVT <: AbstractVector}
#     @unpack dh   = problem
#     @unpack grid = dh
#     @assert getdim(grid) == 3
#     _ncells = getncells(grid)
#     cache.E_ff = zero(SVT(undef,_ncells))
#     cache.frefdata = zero(VVT(undef, _ncells))
#     cache.srefdata = zero(VVT(undef, _ncells))
#     cache.nrefdata = zero(VVT(undef, _ncells))
#     cache.fdata = zero(VVT(undef, _ncells))
#     cache.sdata = zero(VVT(undef, _ncells))
#     cache.ndata = zero(VVT(undef, _ncells))
#     helixangledata = zero(SVT(undef,_ncells))
#     helixanglerefdata = zero(SVT(undef,_ncells))
# end

# mutable struct VolumePostProcessorCache{SVT <: AbstractVector}
#     detF_per_cell::SVT
# end

# function prepare_postprocessor_cache!(cache::VolumePostProcessorCache{SVT}, problem, t, uₙ) where {SVT <: AbstractVector}
#     @unpack dh = problem
#     grid = getgrid(dh)
#     _ncells = getncells(grid)
#     cache.detF_per_cell = zero(SVT(undef, _ncells))
# end

# struct StandardIOPostProcessor{IO, CV, CL}
#     io::IO
#     cv::CV
#     subdomains::Vector{Int}
#     # cache_list::CL
# end

# TODO refactor this one into the framework code and put a nice abstraction layer around it
struct StandardMechanicalIOPostProcessor{IO, CV}
    io::IO
    cv::CV
    subdomains::Vector{Int}
end

function (postproc::StandardMechanicalIOPostProcessor)(t, problem, solver_cache)
    @unpack dh = problem
    grid = get_grid(dh)
    @unpack io, cv, subdomains = postproc

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
        field_idx = find_field(sdh, :displacement)
        field_idx === nothing && continue 
        for cell ∈ CellIterator(sdh)
            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            field_dofs  = dof_range(sdh, field_idx)
            uₑ = solver_cache.uₙ[global_dofs] # element dofs

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
                f₀,s₀,n₀ = evaluate_coefficient(problem.constitutive_model.microstructure_model, cell, qp, time)

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
    Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₙ, :displacement)
    # TODo replace by "dump coefficient" function
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

    # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
    # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
end

function solve_test_ring(name_base, constitutive_model, grid, face_models::FM, ip_mech::IPM, ip_geo::IPG, intorder::Int, Δt = 100.0, T = 1000.0) where {ref_shape, IPM <: Interpolation{ref_shape}, IPG <: Interpolation{ref_shape}, FM}
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    problem = semidiscretize(
        StructuralModel(constitutive_model, face_models),
        FiniteElementDiscretization(
            Dict(:displacement => LagrangeCollection{1}()^3),
            [Dirichlet(:displacement, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])],
            # [Dirichlet(:displacement, getfaceset(grid, "left"), (x,t) -> [0.0], [1]),Dirichlet(:displacement, getfaceset(grid, "front"), (x,t) -> [0.0], [2]),Dirichlet(:displacement, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3]), Dirichlet(:displacement, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])],
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValues(QuadratureRule{ref_shape}(intorder-1), ip_mech, ip_geo)
    standard_postproc = StandardMechanicalIOPostProcessor(io, cv_post, [1])

    # Create sparse matrix and residual vector
    solver = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100))

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        standard_postproc
    )
end


# for (filename, ref_shape, order) ∈ [
#     # ("MidVentricularSectionQuadTet.msh", RefTetrahedron, 2),
#     ("MidVentricularSectionTet.msh", RefTetrahedron, 1),
#     # ("MidVentricularSectionHex.msh", RefCube, 1),
#     # ("MidVentricularSectionQuadHex.msh", RefCube, 2) # We have to update FerriteGmsh first, because the hex27 translator is missing. See https://github.com/Ferrite-FEM/FerriteGmsh.jl/pull/29
# ]
function run_simulations()

ref_shape = RefHexahedron
order = 1

ip_fiber = Lagrange{ref_shape, order}()
ip_u = Lagrange{ref_shape, order}()^3
ip_geo = Lagrange{ref_shape, order}()

ring_grid = generate_ring_mesh(8,2,2)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
solve_test_ring("Debug",
    ActiveStressModel(
        Guccione1991PassiveModel(),
        Guccione1993ActiveModel(10.0),
        PelceSunLangeveld1995Model(;calcium_field=CalciumHatField()),
        create_simple_fiber_model(ring_cs, ip_fiber, ip_geo,
            endo_helix_angle = deg2rad(0.0),
            epi_helix_angle = deg2rad(0.0),
            endo_transversal_angle = 0.0,
            epi_transversal_angle = 0.0,
            sheetlet_pseudo_angle = deg2rad(0)
        )
    ), ring_grid, 
    [NormalSpringBC(0.01, "Epicardium")],
    ip_u, ip_geo, 2*order,
    100.0
)


activation_data = readdlm("../data/ActivationFunction_1ms.dat", ' ', Float64, '\n')
pressure_data = readdlm("../data/ActivationFunction_1ms.dat", ' ', Float64, '\n')

ring_grid = generate_ring_mesh(8,2,2)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
solve_test_ring("ring-test",
    ActiveStressModel(
        Guccione1991PassiveModel(),
        Guccione1993ActiveModel(10.0),
        PelceSunLangeveld1995Model(;calcium_field=SpatiallyHomogeneousDataField(
            activation_data[:,1],activation_data[:,2]
        )),
        create_simple_fiber_model(ring_cs, ip_fiber, ip_geo,
            endo_helix_angle = deg2rad(0.0),
            epi_helix_angle = deg2rad(0.0),
            endo_transversal_angle = 0.0,
            epi_transversal_angle = 0.0,
            sheetlet_pseudo_angle = deg2rad(0)
        )
    ), ring_grid, 
    [NormalSpringBC(0.01, "Epicardium"), PressureFieldBC(
        SpatiallyHomogeneousDataField(pressure_data[:,1],pressure_data[:,2]),
        "Endocardium"
    )],
    ip_u, ip_geo, 2*order,
    10.0, 800.0
)


return

ring_grid = generate_ring_mesh(50,10,10)
filename = "MidVentricularSectionHexG50-10-10"

# ring_grid = saved_file_to_grid("../data/meshes/ring/" * filename)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
ring_fm = create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(60.0), epi_helix_angle = deg2rad(-60.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0)

passive_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))

# solve_test_ring(filename*"_GHM-HO_AS1_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ),
#     ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
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
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_HO_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(passive_model),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Simple_Pelce",
#     ActiveStressModel(
#         passive_model,
#         SimpleActiveStress(),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Piersanti_Pelce",
#     ActiveStressModel(
#         passive_model,
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HONEW_HONEW_RLRSQ100_Pelce",
#     ExtendedHillModel(
#         NewHolzapfelOgden2009Model(),
#         ActiveMaterialAdapter(NewHolzapfelOgden2009Model(;mpU=NullCompressionPenalty())),
#         RLRSQActiveDeformationGradientModel(0.5),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HONEW_Piersanti_Pelce",
#     ActiveStressModel(
#         NewHolzapfelOgden2009Model(),
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"Vallespin2023-Reproducer",
#     ActiveStressModel(
#         Guccione1991PassiveModel(),
#         Guccione1993ActiveModel(150.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.01, "Epicardium")],
#     CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"Vallespin2023-Ring",
#     ActiveStressModel(
#         Guccione1991PassiveModel(),
#         Guccione1993ActiveModel(10),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs,
#     create_simple_fiber_model(ring_cs, ip_fiber, ip_geo,
#         endo_helix_angle = deg2rad(60.0),
#         epi_helix_angle = deg2rad(-60.0),
#         endo_transversal_angle = 0.0,
#         epi_transversal_angle = 0.0,
#         sheetlet_pseudo_angle = deg2rad(20)
#     ),
#     [NormalSpringBC(0.01, "Epicardium")],
#     CalciumHatField(), ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce_MoulinHelixAngle",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(50.0), epi_helix_angle = deg2rad(-40.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce_MoulinHelixAngle_SA45",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(50.0), epi_helix_angle = deg2rad(-40.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_pseudo_angle = deg2rad(45)),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

end

run_simulations()
