# The bilinear operators of the first EP tutorial, assembled on the device against the same forms
# assembled on the host. This is the gate that says a device assembled `M` and `K` may be combined
# into the backward Euler stage matrix at all: entrywise agreement with the host, and reproducibility
# across sweeps.

function _ep_testbed(dims = (24, 24))
    grid = generate_grid(Quadrilateral, dims, Vec{2}((0.0f0, 0.0f0)), Vec{2}((2.5f0, 2.5f0)))
    dh = DofHandler(grid)
    add!(dh, :φₘ, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    return dh
end

@testset "Bilinear operators, host versus device assembly" begin
    dh  = _ep_testbed()
    qrc = QuadratureRuleCollection(2)
    ctx = TimeIntegrationContext(0.0f0, 0.0f0, 0.0f0)

    # The coloring has to actually split the cells, or the sweep would run as one barrier and never
    # exercise the synchronization the colored scatter relies on.
    @test length(Ferrite.create_coloring(Ferrite.get_grid(dh))) > 1

    integrators = (
        Thunderbolt.BilinearMassIntegrator(ConstantCoefficient(1.0f0), qrc, :φₘ),
        Thunderbolt.BilinearDiffusionIntegrator(
            Thunderbolt.ConductivityToDiffusivityCoefficient(
                ConstantCoefficient(SymmetricTensor{2, 2, Float32}((4.5f-5, 0.0f0, 2.0f-5))),
                ConstantCoefficient(1.0f0),
                ConstantCoefficient(1.0f0),
            ),
            qrc,
            :φₘ,
        ),
    )

    @testset "$(nameof(typeof(integrator)))" for integrator in integrators
        host = Thunderbolt.setup_operator(host_assembly_strategy(), integrator, dh)
        Thunderbolt.update_operator!(host, nothing, ctx)

        device = Thunderbolt.setup_operator(
            device_assembly_strategy(; matrix_type = CuCSC),
            integrator,
            dh,
        )
        @test device.A isa CuCSC
        Thunderbolt.update_operator!(device, nothing, ctx)

        # Both arms sum the same Float32 element contributions into the same pattern; the colored
        # device sweep only differs in the order a shared dof accumulates. Measured below 1e-6
        # relative, which is a few Float32 ulps of the entry magnitudes here and four orders below
        # what a wrong coefficient or a misaligned pattern would produce.
        @test SparseMatrixCSC(device.A) ≈ host.A rtol = 1.0f-5
        @test eltype(device.A) === Float32

        first_run = Array(nonzeros(device.A))
        Thunderbolt.update_operator!(device, nothing, ctx)
        @test first_run == Array(nonzeros(device.A))
    end

    @testset "the solver's matrix type and the assembly device have to agree" begin
        solver = BackwardEulerSolver(
            solution_vector_type = CuVector{Float32},
            system_matrix_type   = CuCSR,
        )
        err = @test_throws ErrorException Thunderbolt.setup_operator(
            device_assembly_strategy(),
            first(integrators),
            solver,
            dh,
        )
        @test occursin("CSC", err.value.msg)

        # Naming the format twice, differently, is the other way the two knobs can disagree.
        mismatched = @test_throws ErrorException Thunderbolt.setup_operator(
            device_assembly_strategy(; matrix_type = CuCSC),
            first(integrators),
            BackwardEulerSolver(
                solution_vector_type = CuVector{Float32},
                system_matrix_type   = SparseMatrixCSC{Float32, Int32},
            ),
            dh,
        )
        @test occursin("same type", mismatched.value.msg)
    end
end

@testset "Device assembled operators share the stage matrix pattern" begin
    dh  = _ep_testbed((8, 8))
    qrc = QuadratureRuleCollection(2)
    ctx = TimeIntegrationContext(0.0f0, 0.0f0, 0.0f0)

    # `_implicit_euler_heat_solver_update_system_matrix!` combines the operators into the stage
    # matrix entrywise through `nonzeros`, and the two are allocated by different routines --
    # `FerriteOperators` over its own sparsity pattern, `create_system_matrix` over Ferrite's. The
    # correspondence they rely on is that those patterns are the same one.
    stage = Thunderbolt.create_system_matrix(CuCSC, dh)
    op = Thunderbolt.setup_operator(
        device_assembly_strategy(; matrix_type = CuCSC),
        Thunderbolt.BilinearMassIntegrator(ConstantCoefficient(1.0f0), qrc, :φₘ),
        dh,
    )
    Thunderbolt.update_operator!(op, nothing, ctx)

    @test Array(stage.colPtr) == Array(op.A.colPtr)
    @test Array(stage.rowVal) == Array(op.A.rowVal)
end
