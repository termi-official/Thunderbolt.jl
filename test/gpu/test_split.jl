# The whole monodomain solve of the first EP tutorial, on the device, in the two shapes a device
# solve comes in: host assembly mirrored into a `CuSparseMatrix`, and assembly on the device itself.
# Both run a device CG for the heat child, a kernel for the reaction child, and the operator
# splitting on `CuVector`s. The host arm is the same problem in the same precision, so what separates
# it from either is the order the reductions run in, nothing else.

function _monodomain_form(;
    n = 32,
    assembly_strategy = Thunderbolt.default_strategy(),
    qrcs = Dict{Symbol, Any}(),
)
    mesh = generate_mesh(Quadrilateral, (n, n), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    ep_model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5))),
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedFHNModel{Float32}(),
        CartesianCoordinateSystem(mesh),
        :φₘ,
        :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(ep_model),
        FiniteElementDiscretization(
            Dict(:φₘ => LagrangeCollection{1}());
            qrcs,
            assembly_strategy,
        ),
        mesh,
    )
end

@testset "Reaction diffusion split, host versus device" begin
    odeform = _monodomain_form()

    u₀ = create_initial_condition(odeform, Float32)
    setvariable!(u₀, odeform, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? 1.0f0 : 0.0f0
    end
    setvariable!(u₀, odeform, :s) do x
        x[2] ≥ 1.25 ? 0.1f0 : 0.0f0
    end

    tspan  = (0.0f0, 5.0f0)
    Δt     = 1.0f0
    nsteps = 5

    function build(form, u0, VT, SpMatType)
        timestepper = LieTrotterGodunov((
            BackwardEulerSolver(
                solution_vector_type = VT,
                system_matrix_type   = SpMatType,
                inner_solver         = KrylovJL_CG(atol = 1.0f-10, rtol = 1.0f-8),
            ),
            AdaptiveForwardEulerSubstepper(solution_vector_type = VT, reaction_threshold = 0.1f0),
        ))
        return init(OperatorSplittingProblem(form, u0, tspan), timestepper; dt = Δt)
    end

    # The device assembly arm needs its own semidiscretization: which device assembles is the model
    # side's `assembly_strategy`, not a solver option, and the matrix type has to be the CSC one
    # Ferrite ships a device assembler for.
    # The element precision is elected on the quadrature collection, the device's `value_type` is
    # the global system's; this arm names both `Float32`.
    devform = _monodomain_form(;
        assembly_strategy = device_assembly_strategy(),
        qrcs = Dict(:φₘ => QuadratureRuleCollection(Float32, 2)),
    )

    cpu = build(odeform, copy(u₀), Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    gpu = build(odeform, CuVector(u₀), CuVector{Float32}, CuCSR)
    gpu_assembled = build(devform, CuVector(u₀), CuVector{Float32}, CuCSC)

    # The heat child addresses a contiguous stretch of the state, so on the device it gets a plain
    # `CuArray` over the parent's storage rather than a gather through an uploaded index vector.
    @test odeform.solution_indices[1] isa AbstractUnitRange
    @test gpu.child_subintegrators[1].u isa CuVector{Float32}
    @test gpu.child_subintegrators[2].u isa CuVector{Float32}

    # Host assembly mirrors, device assembly does not: the operator owns the device matrix outright.
    @test gpu.child_subintegrators[1].cache.stage.M isa Thunderbolt.MirroredBilinearOperator
    let stage = gpu_assembled.child_subintegrators[1].cache.stage
        @test stage.M isa Thunderbolt.BilinearFerriteOperator
        @test stage.M.A isa CuCSC
        @test stage.K.A isa CuCSC
    end

    φₘ = solution_variable(odeform, :φₘ)
    for _ = 1:nsteps
        step!(cpu)
        step!(gpu)
        step!(gpu_assembled)
        @test gpu.t == cpu.t
        @test gpu_assembled.t == cpu.t
        # Both arms run the same Float32 arithmetic and both CGs converge to the same tolerance, so
        # what separates them is the order the two implementations reduce in: measured at 8e-8
        # relative over these five steps, a few Float32 ulps. The tolerance leaves two orders of
        # magnitude of headroom for a different CUSPARSE reduction order and is still four orders
        # below what a stale mirror or a misordered `nonzeros` would produce. The device assembled
        # arm adds the colored sweep's accumulation order to that, one more ulp scale difference.
        @test getvariable(Array(gpu.u), φₘ) ≈ getvariable(cpu.u, φₘ) rtol = 1.0f-5
        @test getvariable(Array(gpu_assembled.u), φₘ) ≈ getvariable(cpu.u, φₘ) rtol = 1.0f-5
    end
    @test Array(gpu.u) ≈ cpu.u rtol = 1.0f-5
    @test Array(gpu_assembled.u) ≈ cpu.u rtol = 1.0f-5
    # The wave moved, so the agreement above is not two copies of the initial condition.
    @test cpu.u ≉ u₀
end
