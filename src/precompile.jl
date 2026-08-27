# One cold solve per solver family the package ships, on the smallest mesh and step count that still
# reaches every stage of the stack: element assembly, local condensation, the global Newton, the
# linear solve and the time integrator. Precompiling them moves the first-solve inference cost out of
# every user session and into the package image.
#
# Set the `precompile_workload` preference to `false` to skip the workload, e.g. for a fast
# development rebuild:
#
#     using Preferences, UUIDs
#     set_preferences!(UUID("909927c2-98d5-4a67-bba9-79f03a9ad49b"), "precompile_workload" => false)
#
# The workload writes no files, prints nothing (a `NullLogger` swallows the integrator's finalize
# `@info`) and leaves no global state behind. It runs on the strategy `default_strategy()` returns.

using PrecompileTools: @setup_workload, @compile_workload

if Preferences.@load_preference("precompile_workload", true)
    @setup_workload begin
        hex  = generate_mesh(Hexahedron, (1, 1, 1))
        quad = to_mesh(generate_grid(Quadrilateral, (2, 2), Vec{2}((-2.5, -2.5)), Vec{2}((2.5, 2.5))))
        orthotropic_coefficient = ConstantCoefficient(
            OrthotropicMicrostructure(
                Vec((1.0, 0.0, 0.0)),
                Vec((0.0, 1.0, 0.0)),
                Vec((0.0, 0.0, 1.0)),
            ),
        )
        orthotropic_model = OrthotropicMicrostructureModel(
            ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
            ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
            ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
        )

        @compile_workload begin
            Logging.with_logger(Logging.NullLogger()) do
                # Quasi-static hyperelasticity under homotopy continuation, over a Newton with the
                # default (Krylov) inner solver.
                let dbcs = [
                        Dirichlet(:d, getfacetset(hex, "left"), (x, t) -> [0.0], [1]),
                        Dirichlet(:d, getfacetset(hex, "front"), (x, t) -> [0.0], [2]),
                        Dirichlet(:d, getfacetset(hex, "bottom"), (x, t) -> [0.0], [3]),
                        Dirichlet(:d, getfacetset(hex, "right"), (x, t) -> [0.01t], [1]),
                    ]
                    form = semidiscretize(
                        QuasiStaticModel(
                            :d,
                            PK1Model(HolzapfelOgden2009Model(), orthotropic_coefficient),
                        ),
                        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
                        hex,
                    )
                    integrator = init(
                        QuasiStaticProblem(form, (0.0, 1.0)),
                        HomotopyPathSolver(NewtonRaphsonSolver(; max_iter = 10)),
                        dt = 1.0,
                        verbose = false,
                    )
                    solve!(integrator)
                end

                # Backward Euler over a multi-level Newton: the condensed sarcomere internal
                # variables, with a direct inner solver.
                let dbcs = [
                        Dirichlet(:d, getfacetset(hex, "left"), (x, t) -> [0.0], [1]),
                        Dirichlet(:d, getfacetset(hex, "front"), (x, t) -> [0.0], [2]),
                        Dirichlet(:d, getfacetset(hex, "bottom"), (x, t) -> [0.0], [3]),
                        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]),
                    ]
                    form = semidiscretize(
                        QuasiStaticModel(
                            :d,
                            ActiveStressModel(
                                Guccione1991PassiveModel(),
                                SimpleActiveStress(; Tmax = 220e3),
                                CaDrivenInternalSarcomereModel(
                                    RDQ20MFModel(),
                                    ConstantCoefficient(1.0),
                                ),
                                orthotropic_model,
                            ),
                            (),
                        ),
                        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
                        hex,
                    )
                    problem = QuasiStaticProblem(form, (0.0, 1.0))
                    default_initial_condition!(problem.u0, problem.f)
                    integrator = init(
                        problem,
                        BackwardEulerSolver(;
                            inner_solver = MultiLevelNewtonRaphsonSolver(;
                                newton = NewtonRaphsonSolver(
                                    inner_solver = LinearSolve.UMFPACKFactorization(),
                                    max_iter = 10,
                                    tol = 1e-8,
                                ),
                            ),
                        ),
                        dt = 1.0,
                        verbose = false,
                    )
                    solve!(integrator)
                end

                # Monodomain electrophysiology through Lie-Trotter-Godunov operator splitting.
                let cs = CartesianCoordinateSystem(quad)
                    model = MonodomainModel(
                        ConstantCoefficient(1.0),
                        ConstantCoefficient(1.0),
                        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-4, 0.0, 2.0e-4))),
                        AnalyticalTransmembraneStimulationProtocol(
                            AnalyticalCoefficient(
                                (x, t) -> norm(x) < 0.1 && t < 2.0 ? 0.01 : 0.0,
                                cs,
                            ),
                            [SVector((0.0, 2.1))],
                        ),
                        FHNModel(),
                        :φₘ,
                        :s1,
                    )
                    form = semidiscretize(
                        ReactionDiffusionSplit(model),
                        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                        quad,
                    )
                    integrator = init(
                        OS.OperatorSplittingProblem(
                            form,
                            zeros(solution_size(form)),
                            (0.0, 1.0),
                        ),
                        OS.LieTrotterGodunov((BackwardEulerSolver(), ForwardEulerCellSolver())),
                        dt = 1.0,
                        verbose = false,
                    )
                    solve!(integrator)
                end
            end
        end
    end
end
