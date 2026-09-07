# The whole monodomain solve of the first EP tutorial, on the device: host assembly mirrored into a
# `CuSparseMatrix`, a device CG for the heat child, a kernel for the reaction child, and the operator
# splitting itself running on `CuVector`s. The host arm is the same problem in the same precision, so
# what separates the two is the order the two CG implementations reduce in, nothing else.

function _monodomain_form(; n = 32)
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
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
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

    function build(u0, VT, SpMatType)
        timestepper = LieTrotterGodunov((
            BackwardEulerSolver(
                solution_vector_type = VT,
                system_matrix_type   = SpMatType,
                inner_solver         = KrylovJL_CG(atol = 1.0f-10, rtol = 1.0f-8),
            ),
            AdaptiveForwardEulerSubstepper(solution_vector_type = VT, reaction_threshold = 0.1f0),
        ))
        return init(OperatorSplittingProblem(odeform, u0, tspan), timestepper; dt = Δt)
    end

    cpu = build(copy(u₀), Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    gpu = build(CuVector(u₀), CuVector{Float32}, CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32})

    # The heat child addresses a contiguous stretch of the state, so on the device it gets a plain
    # `CuArray` over the parent's storage rather than a gather through an uploaded index vector.
    @test odeform.solution_indices[1] isa AbstractUnitRange
    @test gpu.child_subintegrators[1].u isa CuVector{Float32}
    @test gpu.child_subintegrators[2].u isa CuVector{Float32}

    φₘ = solution_variable(odeform, :φₘ)
    for _ = 1:nsteps
        step!(cpu)
        step!(gpu)
        @test gpu.t == cpu.t
        # Both arms run the same Float32 arithmetic and both CGs converge to the same tolerance, so
        # what separates them is the order the two implementations reduce in: measured at 8e-8
        # relative over these five steps, a few Float32 ulps. The tolerance leaves two orders of
        # magnitude of headroom for a different CUSPARSE reduction order and is still four orders
        # below what a stale mirror or a misordered `nonzeros` would produce.
        @test getvariable(Array(gpu.u), φₘ) ≈ getvariable(cpu.u, φₘ) rtol = 1.0f-5
    end
    @test Array(gpu.u) ≈ cpu.u rtol = 1.0f-5
    # The wave moved, so the agreement above is not two copies of the initial condition.
    @test cpu.u ≉ u₀
end
