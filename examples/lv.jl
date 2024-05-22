using Thunderbolt, FerriteGmsh, UnPack

# TODO refactor this one into the framework code and put a nice abstraction layer around it
struct StandardMechanicalIOPostProcessor2{IO, CVC, CSC}
    io::IO
    cvc::CVC
    csc::CSC
end

function (postproc::StandardMechanicalIOPostProcessor2)(t, problem::Thunderbolt.SplitProblem, solver_cache)
    (postproc::StandardMechanicalIOPostProcessor2)(t, problem.A, solver_cache.A_solver_cache)
    (postproc::StandardMechanicalIOPostProcessor2)(t, problem.B, solver_cache.B_solver_cache)
end


function (postproc::StandardMechanicalIOPostProcessor2)(t, problem::Thunderbolt.ODEProblem, solver_cache)
    @show solver_cache.uₙ
end

function (postproc::StandardMechanicalIOPostProcessor2)(t, problem::Thunderbolt.RSAFDQ20223DProblem, solver_cache)
    @unpack dh, constitutive_model = problem.structural_problem
    grid = Ferrite.get_grid(dh)
    @unpack io, cvc, csc = postproc

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
    for sdh ∈ dh.subdofhandlers
        field_idx = Ferrite.find_field(sdh, :displacement)
        field_idx === nothing && continue 
        cv = getcellvalues(cvc, dh.grid.cells[first(sdh.cellset)])
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
                f₀,s₀,n₀ = evaluate_coefficient(constitutive_model.microstructure_model, cell, qp, time)

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
    # Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₙ, :displacement)
    Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₙ.blocks[1], :displacement)
    # Thunderbolt.store_timestep_field!(io, t, coordinate_system.dh, coordinate_system.u_transmural, "transmural")
    # Thunderbolt.store_timestep_field!(io, t, coordinate_system.dh, coordinate_system.u_apicobasal, "apicobasal")
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

    @show Thunderbolt.compute_chamber_volume(dh, solver_cache.uₙ, "Endocardium", problem.tying_problem.chambers[1])
    # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
    # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
end


function solve_ideal_lv(name_base, constitutive_model, grid, coordinate_system, face_models, ip_mech::Thunderbolt.VectorInterpolationCollection, qr_collection::QuadratureRuleCollection, Δt, T = 1000.0)
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    problem = semidiscretize(
        # StructuralModel(:displacement, constitutive_model, face_models),
        RSAFDQ2022Split(RSAFDQ2022Model(
                StructuralModel(:displacement, constitutive_model, face_models),
                Thunderbolt.DummyLumpedCircuitModel(t->9.25),
                # RSAFDQ2022LumpedCicuitModel{Float64, Float64,Float64,Float64,Float64,Float64}(),
                LumpedFluidSolidCoupler(
                    [
                        ChamberVolumeCoupling(
                            "Endocardium",
                            RSAFDQ2022SurrogateVolume(),
                            :Vₗᵥ
                        )
                    ],
                    :displacement
                )
            )
        ),
        FiniteElementDiscretization(
            Dict(:displacement => ip_mech),
            # Dirichlet[],
            [Dirichlet(:displacement, getfaceset(grid, "Base"), (x,t) -> [0.0], [3])],
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValueCollection(qr_collection, ip_mech)
    standard_postproc = StandardMechanicalIOPostProcessor2(io, cv_post, CoordinateSystemCoefficient(coordinate_system))

    # Create sparse matrix and residual vector
    solver = LTGOSSolver(
        LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100, tol=1e-2)),
        ForwardEulerSolver(10),
    )

    # solver = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=20))

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        standard_postproc
    )
end

scaling_factor = 1.6
LV_grid = Thunderbolt.hexahedralize(Thunderbolt.generate_ideal_lv_mesh(15,2,7; inner_radius = Float64(scaling_factor*0.7), outer_radius = Float64(scaling_factor*1.0), longitudinal_upper = Float64(scaling_factor*0.2), apex_inner = Float64(scaling_factor*1.3), apex_outer = Float64(scaling_factor*1.5)))
# LV_grid = Thunderbolt.generate_ring_mesh(16,4,4)
order = 1
intorder = max(2*order-1,2)
ip_u = LagrangeCollection{order}()^3
qr_u = QuadratureRuleCollection(intorder-1)
LV_cs = compute_LV_coordinate_system(LV_grid)
# LV_cs = compute_midmyocardial_section_coordinate_system(LV_grid)
LV_fm = create_simple_microstructure_model(LV_cs, LagrangeCollection{1}()^3, endo_helix_angle = deg2rad(-60.0), epi_helix_angle = deg2rad(70.0), endo_transversal_angle = deg2rad(10.0), epi_transversal_angle = deg2rad(-20.0))
passive_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(100.0))
# passive_model = Guccione1991PassiveModel(;C₀ = 3e1, Bᶠᶠ = 8.0, Bˢˢ = 6.0, Bⁿⁿ = 3.0, Bᶠˢ = 12.0, Bⁿˢ = 3.0, Bᶠⁿ = 3.0, mpU = SimpleCompressionPenalty(0.8e3))

function calcium_profile_function(x,t)
    t/1000.0 < 0.5 ? (1-x.transmural*0.7)*2.0*t/1000.0 : (2.0-2.0*t/1000.0)*(1-x.transmural*0.7)
end

function pressure_field_fun(x,t)
    1*sin(π*t/1000)
end

pressure_field = AnalyticalCoefficient(
    pressure_field_fun,
    CoordinateSystemCoefficient(LV_cs)
)

integral_bcs = (NormalSpringBC(1.0, "Epicardium"),)
# integral_bcs = (NormalSpringBC(0.1, "Epicardium"),NormalSpringBC(0.1, "Base"))
# integral_bcs = (NormalSpringBC(1.0, "Epicardium"),PressureFieldBC(pressure_field, "Endocardium"))
# integral_bcs = (RobinBC(1.0, "Epicardium"),PressureFieldBC(pressure_field, "Endocardium"))


using Thunderbolt.TimerOutputs
TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
solve_ideal_lv("lv_test",
    ActiveStressModel(
        Guccione1991PassiveModel(),
        PiersantiActiveStress(;Tmax=10.0),
        PelceSunLangeveld1995Model(;calcium_field=AnalyticalCoefficient(
            calcium_profile_function,
            CoordinateSystemCoefficient(LV_cs)
        )),
        LV_fm,
    ), LV_grid, LV_cs,
    integral_bcs,
    ip_u, qr_u, 25.0, 1000.0
)
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)
