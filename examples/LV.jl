# TODO FIXME
#    1. hexahedralize does not transfer boundary tags correctly
#    2. this example should show how to couple with a 0D fluid
using Thunderbolt, FerriteGmsh, UnPack

# TODO refactor this one into the framework code and put a nice abstraction layer around it
struct StandardMechanicalIOPostProcessor2{IO, CV, CC}
    io::IO
    cv::CV
    subdomains::Vector{Int}
    coordinate_system::CC
end

(postproc::StandardMechanicalIOPostProcessor2)(t, problem::Thunderbolt.SplitProblem, solver_cache) = (postproc::StandardMechanicalIOPostProcessor2)(t, problem.A.base_problems[1], solver_cache.A_solver_cache)

function (postproc::StandardMechanicalIOPostProcessor2)(t, problem, solver_cache)
    @unpack dh = problem
    grid = Ferrite.get_grid(dh)
    @unpack io, cv, subdomains, coordinate_system = postproc

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
            for qp in QuadratureIterator(cv)
                dΩ = getdetJdV(cv, qp)

                # Compute deformation gradient F
                ∇u = function_gradient(cv, qp, uₑ)
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
                x_global = spatial_coordinate(cv, qp, coords)

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
    Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₙ[Block(1)], :displacement)
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

    @show Thunderbolt.compute_chamber_volume(dh, solver_cache.uₙ, "Endocardium", Thunderbolt.Hirschvogel2016SurrogateVolume())
    # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
    # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
end


function solve_ideal_lv(name_base, constitutive_model, grid, coordinate_system, face_models, ip_collection::Thunderbolt.ScalarInterpolationCollection, ip_mech::IPM, ip_geo::IPG, intorder::Int, Δt = 100.0, T = 1000.0) where {ref_shape, IPM <: Interpolation{ref_shape}, IPG <: Interpolation{ref_shape}}
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    problem = semidiscretize(
        ReggazoniSalvadorAfricaSplit(CoupledModel(
            [
                StructuralModel(constitutive_model, face_models),
                ReggazoniSalvadorAfricaLumpedCicuitModel{Float64,Float64,Float64,Float64,Float64}()
            ],
            [
                Coupling(1,2,LumpedFluidSolidCoupler(
                    Hirschvogel2016SurrogateVolume()
                ))
            ]
        )),
        FiniteElementDiscretization(
            Dict(:displacement => ip_collection^3),
            Dirichlet[],
            # [Dirichlet(:displacement, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])],
            # [Dirichlet(:displacement, getfaceset(grid, "left"), (x,t) -> [0.0], [1]),Dirichlet(:displacement, getfaceset(grid, "front"), (x,t) -> [0.0], [2]),Dirichlet(:displacement, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3]), Dirichlet(:displacement, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])],
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValues(QuadratureRule{ref_shape}(intorder-1), ip_mech, ip_geo)
    standard_postproc = StandardMechanicalIOPostProcessor2(io, cv_post, [1], coordinate_system)

    # Create sparse matrix and residual vector
    solver = LTGOSSolver(
        LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100)),
        ForwardEulerSolver(),
    )

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        standard_postproc
    )
end

LV_grid = Thunderbolt.hexahedralize(Thunderbolt.generate_ideal_lv_mesh(15,2,6))
ref_shape = RefHexahedron
order = 1
ip_collection = LagrangeCollection{order}()
ip = getinterpolation(ip_collection^3, ref_shape)
ip_fiber = getinterpolation(ip_collection, ref_shape)
ip_geo = getinterpolation(ip_collection, ref_shape)
LV_cs = compute_LV_coordinate_system(LV_grid, ip)
LV_fm = create_simple_fiber_model(LV_cs, ip_fiber, ip_geo, endo_helix_angle = -60.0, epi_helix_angle = 70.0, endo_transversal_angle = 10.0, epi_transversal_angle = -20.0)
passive_ho_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))
solve_ideal_lv("lv_test",
    ActiveStressModel(
        passive_ho_model,
        SimpleActiveStress(10.0),
        PelceSunLangeveld1995Model(calcium_field=CalciumHatField()),
        LV_fm,
    ), LV_grid, LV_cs,
    [NormalSpringBC(0.001, "Epicardium")],
    ip_collection,
    ip, ip_geo, max(2*order-1,2)
)
