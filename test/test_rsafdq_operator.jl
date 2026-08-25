using Test, Thunderbolt
using Ferrite, LinearAlgebra, SparseArrays
using BlockArrays
include(joinpath(@__DIR__, "testfixtures.jl"))

# Equivalence harness for the 3D-0D coupled operator.
#
# `test/integration/test_fsi.jl` asserts only that the coupled solve converges, which a wrong
# Jacobian or a dropped coupling term can survive. This file pins the assembled `(J, r)` of the
# single-chamber ideal-LV model at a fixed non-solution state against a stored reference, so that
# changes to how the operator is built have to reproduce the numbers entry for entry.

const RSAFDQ_REFERENCE_FILE = joinpath(@__DIR__, "data", "rsafdq_operator_reference.jl")

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
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs,
            # Pinned reference: deterministic summation needs the sequential device.
            assembly_strategy = Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
        ),
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
    row, col, val = findnz(sparse(J))
    pressure_dofs = [c.pressure_dof_index for c in state.f.tying_info.chambers]
    n_field_dofs = ndofs(state.f.structural_function.dh) - length(state.f.tying_info.chambers)
    mkpath(dirname(path))
    # A plain-text pin rather than a binary artifact: `repr` round-trips every `Float64` bit-exactly,
    # so this is not a rounded copy of the tree that produced it. `read_rsafdq_reference` below
    # `include`s this file for its value; its own field names (`row`/`col`/`val`/`n`) are the sparse
    # `J`'s triplet form, kept apart from `J` itself to avoid shadowing it.
    open(path, "w") do io
        println(io, "# Auto-generated by `write_rsafdq_reference` in test/test_rsafdq_operator.jl.")
        println(io, "# Do not hand-edit; regenerate from a tree whose assembled values are known good.")
        println(io, "(;")
        println(io, "    n             = ", size(J, 1), ",")
        println(io, "    row           = ", repr(row), ",")
        println(io, "    col           = ", repr(col), ",")
        println(io, "    val           = ", repr(val), ",")
        println(io, "    r             = ", repr(r), ",")
        println(io, "    pressure_dofs = ", repr(pressure_dofs), ",")
        println(io, "    n_field_dofs  = ", n_field_dofs, ",")
        println(io, ")")
    end
    return path
end

function read_rsafdq_reference(path = RSAFDQ_REFERENCE_FILE)
    raw = include(path)
    return (;
        J             = sparse(raw.row, raw.col, raw.val, raw.n, raw.n),
        r             = raw.r,
        pressure_dofs = raw.pressure_dofs,
        n_field_dofs  = raw.n_field_dofs,
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

"""
    two_chamber_state(; seed = 7, device = SequentialCPUDevice())

A passive left heart with one [`Thunderbolt.Pressure3D0DVolumeCoupler`](@ref) per chamber, its
operator, and a fixed pseudo-random state to assemble at.

The couplers tie over the *closed* chamber surfaces, endocardium plus the matching face of the
valvular plate, which is what makes their volume integral the enclosed volume. The two chamber
pressures are the only unknowns outside the mesh, and they exist because the two facet terms say
so -- nothing here declares them a second time. The state is deliberately not a solution, for the
reason given at [`rsafdq_reference_state`](@ref).

`device` defaults to the sequential device, deterministic summation for a pinned reference; pass
`PolyesterDevice()` to build the same problem for the threaded-vs-sequential equivalence check.

Returns `(; f, op, u, pressure_symbols, pressure_dofs, n_u, models)`.
"""
function two_chamber_state(; seed = 7, device = Thunderbolt.SequentialCPUDevice())
    scaling_factor = 3.9
    mesh = generate_ideal_lh_mesh(
        6, 1, 2, 2;
        inner_radius       = scaling_factor * 0.7,
        outer_radius       = scaling_factor * 1.0,
        longitudinal_upper = 0.4,
        apex_inner         = scaling_factor * 1.3,
        apex_outer         = scaling_factor * 1.5,
    )

    # The coordinate system is ventricular -- the atrium has no long axis of its own -- so it is
    # built on that subdomain alone, off the ventricular surfaces, with the interior annulus sheet
    # standing in for the basal one. The atrium carries a constant frame instead.
    cs = compute_lv_coordinate_system(
        mesh;
        subdomains       = ["ventricle"],
        axes             = compute_lv_axes(mesh; base = "MitralAnnulus", apex = "Apex"),
        base_name        = "MitralAnnulus",
        endocardium_name = "LVEndocardium",
        epicardium_name  = "LVEpicardium",
    )
    ventricle_microstructure = create_microstructure_model(
        cs,
        LagrangeCollection{1}()^3,
        ODB25LTMicrostructureParameters(αendo = deg2rad(80.0), αepi = deg2rad(-65.0));
        subdomains = ["ventricle"],
    )
    atrium_microstructure = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )

    pressure_symbols = (:pₗᵥ, :pₗₐ)
    couplers = (
        Pressure3D0DVolumeCoupler(
            "LVChamberSurface",
            :d,
            pressure_symbols[1],
            RSAFDQ2022SurrogateVolume(),
        ),
        Pressure3D0DVolumeCoupler(
            "LAChamberSurface",
            :d,
            pressure_symbols[2],
            RSAFDQ2022SurrogateVolume(),
        ),
    )
    # The plate is a carrier, not tissue: passive, three orders of magnitude below the wall and
    # nearly free to change volume, so it follows the annulus without stiffening it.
    plate_material = PK1Model(
        Guccione1991PassiveModel(; C₀ = 1.0e-3, mpU = SimpleCompressionPenalty(5.0e-2)),
        atrium_microstructure,
    )
    # All three subdomains carry both couplers: the declarations have to agree across a domain
    # split, and each coupler only ever traverses the facets of its own chamber. The plate is not
    # along for the ride here -- both of its faces are tying facets, so its own model is what
    # declares them, and the pressure of a closed cavity does act on it.
    models = Dict(
        "ventricle" => QuasiStaticModel(
            :d,
            PK1Model(Guccione1991PassiveModel(), ventricle_microstructure),
            couplers,
        ),
        "atrium" => QuasiStaticModel(
            :d,
            PK1Model(Guccione1991PassiveModel(), atrium_microstructure),
            couplers,
        ),
        "valvular-plane" => QuasiStaticModel(:d, plate_material, couplers),
    )

    dbcs = [
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor1"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor2"), (x, t) -> (0.0, 0.0), [2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor3"), (x, t) -> (0.0,), [3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor4"), (x, t) -> (0.0,), [3]),
    ]
    # The default facet rule is exact for this bilinear-geometry volume functional, and the chamber
    # volume is the enclosed volume only where the surface is closed *and* the rule is exact.
    f = semidiscretize(
        models,
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs,
            # Pinned reference: deterministic summation needs the sequential device (the default
            # here); the threaded-vs-sequential equivalence test overrides it via `device`.
            assembly_strategy = Thunderbolt.SequentialAssemblyStrategy(device),
        ),
        mesh,
    )

    dh = f.dh
    pressure_dofs = [only(algebraic_dofs(dh, sym)) for sym in pressure_symbols]
    n_u = ndofs(dh) - length(pressure_symbols)

    # Both pressures sit in the tail of every element-local system, so both need the sparsity of a
    # `CellCoupling` over the whole handler -- see `Thunderbolt._chamber_coupling`.
    all_cells = collect(Int, Iterators.flatten(sdh.cellset for sdh in dh.subdofhandlers))
    couplings = Tuple(
        Thunderbolt.FerriteOperators.CellCoupling(
            all_cells;
            algebraic_coupling = ((:d, sym),),
        ) for sym in pressure_symbols
    )
    strategy = Thunderbolt.AssemblyStrategy(
        Thunderbolt.FullAssembly(
            Thunderbolt.FerriteOperators.StandardOperatorSpecification(;
                algebraic_couplings = couplings,
                constraint_handler = f.ch,
            ),
        ),
        Thunderbolt.SequentialScheduling(),
        Thunderbolt.get_strategy(f).device,
    )
    op = Thunderbolt.setup_operator(
        strategy,
        f.integrator,
        dh;
        slots = Thunderbolt.THUNDERBOLT_STAGE_SLOTS,
    )

    u = 0.05 .* (_pinned_uniform(Thunderbolt.solution_size(f), seed) .- 0.5)
    u[pressure_dofs] .= RSAFDQ_REFERENCE_PRESSURE

    return (; f, op, u, pressure_symbols, pressure_dofs, n_u, models)
end

"""
    deformed_coordinates(dh, u)

The grid's nodes moved by the displacement field in `u`.

With a linear geometry and a linear displacement interpolation the deformed mesh is exactly the mesh
with moved nodes, so the deformed chamber volume follows from node positions alone -- independently
of how the operator computes it.
"""
function deformed_coordinates(dh, u)
    grid = dh.grid
    displacements = zeros(Vec{3, Float64}, getnnodes(grid))
    for sdh in dh.subdofhandlers
        displacement_range = dof_range(sdh, :d)
        for cellid in sdh.cellset
            dofs = celldofs(dh, cellid)[displacement_range]
            for (i, nodeid) in enumerate(getcells(grid, cellid).nodes)
                displacements[nodeid] = Vec(ntuple(c -> u[dofs[3*(i-1)+c]], 3))
            end
        end
    end
    return [
        Ferrite.get_node_coordinate(node) + displacements[i] for
        (i, node) in enumerate(getnodes(grid))
    ]
end

@testset "Two-chamber 3D-0D operator" begin
    state = two_chamber_state()
    dh = state.f.dh
    integrator = state.f.integrator
    n_chambers = length(state.pressure_symbols)

    @testset "declared unknowns" begin
        # The couplers own the pressures, so the model derives them rather than being told.
        for model in values(state.models)
            @test collect(Thunderbolt.algebraic_variables(model)) ==
                  collect(state.pressure_symbols)
        end
        @test dh.algebraic_names == collect(state.pressure_symbols)
        @test state.pressure_dofs == state.n_u .+ (1:n_chambers)
        # Every subdomain sees the same tail, in the same order.
        for sdh in dh.subdofhandlers
            @test Thunderbolt.FerriteOperators.global_dofs(integrator, sdh) == state.pressure_dofs
        end
        @test Thunderbolt.FerriteOperators.algebraic_items(integrator, dh) ==
              [[dof] for dof in state.pressure_dofs]
    end

    @testset "facet coverage" begin
        declared = [
            facet for sdh in dh.subdofhandlers for
            facet in Thunderbolt.FerriteOperators.facet_items(integrator, sdh)
        ]
        surfaces = union(
            getfacetset(dh.grid, "LVChamberSurface"),
            getfacetset(dh.grid, "LAChamberSurface"),
        )
        @test length(declared) == length(surfaces)
        @test Set(declared) == Set(surfaces)
        for name in ("LVEndocardium", "LAEndocardium", "LVValvularPlane", "LAValvularPlane")
            @test Set(getfacetset(dh.grid, name)) ⊆ Set(declared)
        end

        # `facet_items` filters per subdomain, so the plate's faces reach the operator only because
        # the plate's own model declares them. Its wedges and its hexahedra are handled apart, so
        # the subdomain is several subhandlers.
        plate_cells = getcellset(dh.grid, "valvular-plane")
        plate_sdhs = [sdh for sdh in dh.subdofhandlers if all(∈(plate_cells), sdh.cellset)]
        @test !isempty(plate_sdhs)
        plate_declared = [
            facet for sdh in plate_sdhs for
            facet in Thunderbolt.FerriteOperators.facet_items(integrator, sdh)
        ]
        @test !isempty(plate_declared)
        @test Set(plate_declared) == union(
            getfacetset(dh.grid, "LVValvularPlane"),
            getfacetset(dh.grid, "LAValvularPlane"),
        )
    end

    V⁰ᴰ = [120.0, 60.0]
    ctx = Thunderbolt.TimeIntegrationContext(RSAFDQ_REFERENCE_T, 0.0, 0.0)
    J, r = assemble_pair(state.op, state.u, (V⁰ᴰ = V⁰ᴰ,), ctx)

    @testset "both chambers are exercised" begin
        @test size(J) == (state.n_u + n_chambers, state.n_u + n_chambers)
        for dof in state.pressure_dofs
            @test maximum(abs, J[dof, :]) > 0
            @test maximum(abs, J[:, dof]) > 0
            @test abs(r[dof]) > 0
        end
    end

    @testset "each chamber row is its closed volume" begin
        # `RSAFDQ2022SurrogateVolume` measures `-∮ (x + d - b) ⋅ ĥ n̂ dΓ` along one axis, which is
        # the enclosed volume only on a closed surface -- and then it is the same along every axis,
        # and independent of `b`. The plate is meshed, so the closure moves with the wall and the
        # deformed surface is still closed.
        deformed = deformed_coordinates(dh, state.u)
        for (i, name) in enumerate(("LVChamberSurface", "LAChamberSurface"))
            volumes = surface_volumes(dh.grid, getfacetset(dh.grid, name), deformed)
            @test volumes[1] ≈ volumes[2] rtol = 1.0e-10
            @test volumes[1] ≈ volumes[3] rtol = 1.0e-10
            @test r[state.pressure_dofs[i]] ≈ -volumes[1] - V⁰ᴰ[i] rtol = 1.0e-10
        end
    end

    @testset "V⁰ᴰ enters each chamber row alone" begin
        # `r[p] -= V⁰ᴰ` is the whole dependence, so swapping the two reference volumes has to move
        # the two chamber entries by the difference and nothing else at all.
        swapped = reverse(V⁰ᴰ)
        J′, r′ = assemble_pair(state.op, state.u, (V⁰ᴰ = swapped,), ctx)
        Δ = r′ - r
        @test Δ[state.pressure_dofs] ≈ V⁰ᴰ - swapped rtol = 1.0e-12
        others = setdiff(1:length(r), state.pressure_dofs)
        @test maximum(abs, Δ[others]) ≈ 0 atol = 1.0e-12 * maximum(abs, r)
        @test approx_entrywise(J′, J)
    end
end

@testset "Threaded vs sequential two-chamber equivalence" begin
    # The first genuine multi-worker exercise of the full coupled operator (cells + facet items +
    # algebraic items + global dofs): the atomic scatter under `PolyesterDevice` has to reproduce
    # the sequential assembly, entry for entry up to summation order (never `==`, see
    # `approx_entrywise`).
    seq = two_chamber_state()
    par = two_chamber_state(device = Thunderbolt.PolyesterDevice())

    V⁰ᴰ = [120.0, 60.0]
    ctx = Thunderbolt.TimeIntegrationContext(RSAFDQ_REFERENCE_T, 0.0, 0.0)
    J_seq, r_seq = assemble_pair(seq.op, seq.u, (V⁰ᴰ = V⁰ᴰ,), ctx)
    J_par, r_par = assemble_pair(par.op, par.u, (V⁰ᴰ = V⁰ᴰ,), ctx)

    @test approx_entrywise(J_par, J_seq)
    @test approx_entrywise(r_par, r_seq)
end
