using Thunderbolt

@testset "EP wave propagation" begin
    function simple_initializer!(u₀, f::GenericSplitFunction)
        # TODO cleaner implementation. We need to extract this from the types or via dispatch.
        heatfun = f.functions[1]
        heat_dofrange = f.dof_ranges[1]
        odefun = f.functions[2]
        ionic_model = odefun.ode

        ϕ₀ = @view u₀[heat_dofrange];
        # TODO extraction these via utility functions
        dh = heatfun.dh
        for sdh in dh.subdofhandlers
            for cell in CellIterator(sdh)
                _celldofs = celldofs(cell)
                φₘ_celldofs = _celldofs[dof_range(sdh, :φₘ)]
                # TODO query coordinate directly from the cell model
                coordinates = getcoordinates(cell)
                for (i, x) in zip(φₘ_celldofs,coordinates)
                    ϕ₀[i] = norm(x)/2
                end
            end
        end
    end

    function solve_waveprop(mesh, coeff, subdomains, isadaptive = false)
        cs = CoordinateSystemCoefficient(CartesianCoordinateSystem(mesh))
        model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            coeff,
            Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                # Stimulate at apex
                AnalyticalCoefficient((x,t) -> norm(x) < 0.25 && t < 2.0 ? 0.5 : 0.0, cs),
                [SVector((0.0, 2.1))]
            ),
            Thunderbolt.FHNModel(),
            :φₘ, :s
        )

        odeform = semidiscretize(
            ReactionDiffusionSplit(model),
            FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}()), Dirichlet[], subdomains),
            mesh
        )

        _timestepper = LieTrotterGodunov((
            BackwardEulerSolver(),
            ForwardEulerCellSolver()
        ))
        if isadaptive
            timestepper = Thunderbolt.AdaptiveOperatorSplittingAlgorithm(
                _timestepper,
                Thunderbolt.ReactionTangentController(0.5, 1.0, (0.01, 0.3))
            )
        else
            timestepper = _timestepper
        end

        u₀ = zeros(Float64, OS.function_size(odeform))
        simple_initializer!(u₀, odeform)

        tspan = (0.0, 10.0)
        problem = OperatorSplittingProblem(odeform, u₀, tspan)

        integrator = DiffEqBase.init(problem, timestepper, dt=1.0, verbose=true)
        DiffEqBase.solve!(integrator)
        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
        @test integrator.u ≉ u₀
        return integrator.u
    end

    mesh  = generate_mesh(Hexahedron, (4, 4, 4), Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,1.0,1.0)))
    coeff = ConstantCoefficient(SymmetricTensor{2,3,Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, [""])
    u_adaptive = solve_waveprop(mesh, coeff, [""], true)
    @test u ≈ u_adaptive

    mesh  = generate_ideal_lv_mesh(4,1,1)
    coeff = ConstantCoefficient(SymmetricTensor{2,3,Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, [""])
    u_adaptive = solve_waveprop(mesh, coeff, [""], true)
    @test u ≈ u_adaptive
    
    mesh = to_mesh(generate_mixed_grid_2D())
    coeff = ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)))
    u = solve_waveprop(mesh, coeff, ["Pacemaker", "Myocardium"])
    u_adaptive = solve_waveprop(mesh, coeff, ["Pacemaker", "Myocardium"], true)
    @test u ≈ u_adaptive
    u = solve_waveprop(mesh, coeff, ["Pacemaker"])
    u_adaptive = solve_waveprop(mesh, coeff, ["Pacemaker"], true)
    @test u ≈ u_adaptive
    u = solve_waveprop(mesh, coeff, ["Myocardium"])
    u_adaptive = solve_waveprop(mesh, coeff, ["Myocardium"], true)
    @test u ≈ u_adaptive

    mesh = to_mesh(generate_mixed_dimensional_grid_3D())
    coeff = ConstantCoefficient(SymmetricTensor{2,3,Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, ["Ventricle"])
    u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle"])
    @test u ≈ u_adaptive
    coeff = ConstantCoefficient(SymmetricTensor{2,3,Float64}((5e-5, 0, 0, 5e-5, 0, 5e-5)))
    u = solve_waveprop(mesh, coeff, ["Purkinje"])
    u_adaptive = solve_waveprop(mesh, coeff, ["Purkinje"])
    @test u ≈ u_adaptive
    u = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"])
    u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"])
    @test u ≈ u_adaptive
end
