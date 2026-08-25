using Test, Thunderbolt
using Ferrite, LinearAlgebra, SparseArrays, JLD2
using BlockArrays

# Equivalence harness for the 3D-0D coupled operator.
#
# `test/integration/test_fsi.jl` asserts only that the coupled solve converges, which a wrong
# Jacobian or a dropped coupling term can survive. This file pins the assembled `(J, r)` of the
# single-chamber ideal-LV model at a fixed non-solution state against a stored reference, so that
# changes to how the operator is built have to reproduce the numbers entry for entry.

const RSAFDQ_REFERENCE_FILE = joinpath(@__DIR__, "data", "rsafdq_operator_reference.jld2")

# Pinned solver-supplied data. Within one 3D solve `V⁰ᴰ` is a constant, not an unknown; pinning it
# here rather than taking it from a transfer keeps the harness independent of the 0D solve.
const RSAFDQ_REFERENCE_V⁰ᴰ = 120.0
# The pseudo-time the calcium transient is sampled at, and the pinned chamber pressure.
const RSAFDQ_REFERENCE_T = 100.0
const RSAFDQ_REFERENCE_PRESSURE = 1.3

# The state is drawn here rather than from `Random`: the reference is stored, so the sequence has to
# reproduce across Julia versions, which the `rand` stream does not promise. Knuth's LCG, taking the
# top 53 bits of each word.
function _pinned_uniform(n::Int, seed::Integer)
    u = Vector{Float64}(undef, n)
    x = UInt64(seed)
    for i = 1:n
        x = 0x5851f42d4c957f2d * x + 0x14057b7ef767814f
        u[i] = Float64(x >> 11) / 9007199254740992.0
    end
    return u
end

function _rsafdq_reference_calcium(x::LVCoordinate, t_global)
    t = t_global % 800.0
    0.0 ≤ t ≤ 120.0 && return t / 120.0
    t ≤ 272.0 && return (272.0 - t) / 152.0
    return 0.0
end

"""
    rsafdq_reference_state(; seed = 42)

The single-chamber ideal-LV model of `test/integration/test_fsi.jl` on a coarse mesh, its stage
operator, and a fixed pseudo-random state to assemble at.

The state is deliberately *not* a solution: at equilibrium the residual is zero, which makes the
residual half of the comparison vacuous, and the chamber row is only nonzero away from equilibrium.

Returns `(; f, op, u, p, ctx, pressure_symbol, n_chambers)`.
"""
function rsafdq_reference_state(; seed = 42)
    scaling_factor = 3.9
    mesh = generate_ideal_lv_mesh(
        6, 1, 2;
        inner_radius       = scaling_factor * 0.7,
        outer_radius       = scaling_factor * 1.0,
        longitudinal_upper = 0.4,
        apex_inner         = scaling_factor * 1.3,
        apex_outer         = scaling_factor * 1.5,
        with_control_point = true,
    )

    cs = compute_lv_coordinate_system(mesh; subdomains = ["myocardium"])
    microstructure_model = create_microstructure_model(
        cs,
        LagrangeCollection{1}()^3,
        ODB25LTMicrostructureParameters(αendo = deg2rad(80.0), αepi = deg2rad(-65.0));
        subdomains = ["myocardium"],
    )
    constitutive_model = ActiveStressModel(
        Guccione1991PassiveModel(),
        SimpleActiveStress(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            AnalyticalCoefficient(_rsafdq_reference_calcium, cs),
        ),
        microstructure_model,
    )

    dbcs = [
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor1"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor2"), (x, t) -> (0.0, 0.0), [2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor3"), (x, t) -> (0.0,), [3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor4"), (x, t) -> (0.0,), [3]),
    ]
    solid_model = QuasiStaticModel(
        :d,
        constitutive_model,
        (NormalSpringBC(0.1, "Epicardium"), NormalSpringBC(0.1, "Base")),
    )
    coupler = LumpedFluidSolidCoupler(
        [
            ChamberVolumeCoupling(
                "Endocardium",
                "lv-volume-control",
                RSAFDQ2022SurrogateVolume(),
                :Vₗᵥ,
                :pₗᵥ,
                :pₗᵥ,
            ),
        ],
        :d,
    )
    coupled_model = RSAFDQ2022Model(
        Dict("myocardium" => solid_model),
        RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false),
        coupler,
    )

    splitform = semidiscretize(
        RSAFDQ2022Split(coupled_model),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    f = splitform.functions[1]

    solver = HomotopyPathSolver(NewtonRaphsonSolver(; max_iter = 10, tol = 1e-2))
    op = Thunderbolt.setup_stage_operator(f, solver, nothing, RSAFDQ_REFERENCE_T)

    n = Thunderbolt.solution_size(f)
    n_chambers = length(f.tying_info.chambers)
    u = 0.05 .* (_pinned_uniform(n, seed) .- 0.5)
    u[(n - n_chambers + 1):n] .= RSAFDQ_REFERENCE_PRESSURE
    for chamber in f.tying_info.chambers
        chamber.V⁰ᴰval = RSAFDQ_REFERENCE_V⁰ᴰ
    end

    evaluation = Thunderbolt._homotopy_stage_evaluation(f, RSAFDQ_REFERENCE_T)
    return (; f, op, u, p = evaluation.p, ctx = evaluation.ctx, pressure_symbol = :pₗᵥ, n_chambers)
end

"""
    assemble_pair(op, u, p, ctx) -> (Matrix(J), r)

One fused linearization sweep at `u`, as dense matrix and residual.
"""
function assemble_pair(op, u, p, ctx)
    J = Thunderbolt.getJ(op)
    r = zeros(size(J, 1))
    Thunderbolt.update_linearization!(op, r, (u = u,), p, ctx)
    return (Matrix(J), r)
end

# Entry for entry, because the global dof numbering is the same by construction. `≈` rather than
# `==`: the summation order into a dof shared by several facets is not pinned.
function approx_entrywise(A, B; rtol = 1.0e-10)
    scale = maximum(abs, B)
    return all(isapprox.(A, B; rtol, atol = rtol * (scale > 0 ? scale : one(scale))))
end

"""
    write_rsafdq_reference(path = RSAFDQ_REFERENCE_FILE)

Regenerate the pinned reference from the current tree. Run this only when the assembled values are
*intended* to change; otherwise the harness has nothing left to compare against.
"""
function write_rsafdq_reference(path = RSAFDQ_REFERENCE_FILE)
    state = rsafdq_reference_state()
    J, r = assemble_pair(state.op, state.u, state.p, state.ctx)
    mkpath(dirname(path))
    jldsave(
        path;
        J             = sparse(J),
        r,
        pressure_dofs = [c.pressure_dof_index for c in state.f.tying_info.chambers],
        n_field_dofs  = ndofs(state.f.structural_function.dh) - length(state.f.tying_info.chambers),
    )
    return path
end

read_rsafdq_reference(path = RSAFDQ_REFERENCE_FILE) = jldopen(path, "r") do file
    (;
        J             = file["J"],
        r             = file["r"],
        pressure_dofs = file["pressure_dofs"],
        n_field_dofs  = file["n_field_dofs"],
    )
end

@testset "RSAFDQ2022 3D-0D operator" begin
    isfile(RSAFDQ_REFERENCE_FILE) || error(
        "The pinned reference $(RSAFDQ_REFERENCE_FILE) is missing. Regenerate it with " *
        "`write_rsafdq_reference()` from a tree whose assembled values are known good.",
    )
    reference = read_rsafdq_reference()

    state = rsafdq_reference_state()
    dh = state.f.structural_function.dh
    J, r = assemble_pair(state.op, state.u, state.p, state.ctx)

    @testset "dof numbering" begin
        # The chamber pressure dofs are what license the entry-for-entry comparison below: they have
        # to be the pinned `n_field_dofs + i`, and the field dofs have to keep the leading block.
        pressure_dofs = algebraic_dofs(dh, state.pressure_symbol)
        @test pressure_dofs == reference.pressure_dofs
        @test pressure_dofs == reference.n_field_dofs .+ (1:state.n_chambers)
        @test ndofs(dh) - state.n_chambers == reference.n_field_dofs
    end

    @testset "block layout" begin
        # `SchurComplementLinearSolver` factors the (1,1) block directly, which needs `op.J` to be a
        # `BlockMatrix` split as [field dofs | chamber pressures].
        @test state.op.J isa BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}
        @test blocklengths(axes(state.op.J, 1)) == [reference.n_field_dofs, state.n_chambers]
    end

    @testset "facet coverage" begin
        # Every declared tying facet must be assembled by exactly one subdomain. A boundary set
        # spanning several subdomains (apex wedges beside the hexahedra) once lost the facets
        # outside the first facet's subdomain. The declared set *is* the traversal, and a facet
        # whose cell a subdomain does not own is a setup error, so what is left to check here is
        # that the per-subdomain declarations cover the chamber surfaces without duplication.
        integrator = state.f.structural_function.integrator
        declared = [
            facet for sdh in dh.subdofhandlers for
            facet in Thunderbolt.FerriteOperators.facet_items(integrator, sdh)
        ]
        chamber_facets =
            union((Set(chamber.facets) for chamber in state.f.tying_info.chambers)...)
        @test length(declared) == length(chamber_facets)
        @test Set(declared) == chamber_facets
    end

    pdof = only(reference.pressure_dofs)
    @testset "coupling is exercised" begin
        @test maximum(abs, J[pdof, :]) > 0
        @test maximum(abs, J[:, pdof]) > 0
        @test abs(r[pdof]) > 0
    end

    @testset "equivalence" begin
        J₀ = Matrix(reference.J)
        @test size(J) == size(J₀)
        @test length(r) == length(reference.r)
        @test approx_entrywise(J, J₀)
        @test approx_entrywise(r, reference.r)
    end

    @testset "derivatives" begin
        # The independent referee for the analytic tying tangent: central finite differences against
        # the operator's own residual, chamber rows included. The `-V⁰ᴰ` row is constant in `u`, so
        # its block should come out zero on both sides of the comparison.
        result = Thunderbolt.FerriteOperators.check_derivatives(
            state.op,
            (u = state.u,),
            state.p,
            state.ctx,
        )
        @test result.passed
        for (name, check) in pairs(result.checks)
            check.passed || @info "check_derivatives($name)" check
        end
    end
end
