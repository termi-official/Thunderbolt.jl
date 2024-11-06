using Thunderbolt
using Thunderbolt.TimerOutputs

using Thunderbolt.StaticArrays, UnPack

function steady_state_initializer!(u₀, f::GenericSplitFunction)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.dof_ranges[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    # TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    # Should not be reshape but some array of arrays fun
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i ∈ 1:(Thunderbolt.num_states(ionic_model)-1)
        s₀[:, i] .= default_values[i+1]
    end
end

mesh = generate_mesh(Hexahedron, (80,28,12), Vec((0.0,0.0,0.0)), Vec((20.0,7.0,3.0)))
cs = CoordinateSystemCoefficient(CartesianCoordinateSystem(mesh))

tspan = (0.0, 100.0)
dtvis = 1.0
dt₀ = 0.01


κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,3}((
        κ₁, 0, 0,
           κᵣ, 0,
              κᵣ
    ))),
    AnalyticalTransmembraneStimulationProtocol(
        AnalyticalCoefficient((x,t) -> maximum(x) < 1.5 && t < 2.0 ? 0.5 : 0.0, cs),
        [SVector((0.0, 2.1))]
    ),
    Thunderbolt.PCG2019(),
    :φₘ, :s
)


# ip_collection = LagrangeCollection{1}()
ip_collection = DiscontinuousLagrangeCollection{1}()

odeform = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => ip_collection)),
    mesh
)
u₀ = zeros(Float64, OS.function_size(odeform))
steady_state_initializer!(u₀, odeform)

io = ParaViewWriter("spiral-wave-test")

timestepper = Thunderbolt.ReactionTangentController(
    OS.LieTrotterGodunov((
        BackwardEulerSolver(
            solution_vector_type=Vector{Float32},
            system_matrix_type=Thunderbolt.ThreadedSparseMatrixCSR{Float32, Int32},
            inner_solver=LinearSolve.KrylovJL_CG(atol=1.0f-6, rtol=1.0f-5),
        ),
        AdaptiveForwardEulerSubstepper(
            solution_vector_type=Vector{Float32},
            reaction_threshold=0.1f0,
        )
    )),
    0.5, 1.0, (0.01, 0.011)
)

problem = OS.OperatorSplittingProblem(odeform, u₀, tspan)

_integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

step!(_integrator) # precompile for benchmark below

# TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(_integrator, tspan[1]:dtvis:tspan[2])
    dh = odeform.functions[1].dh
    φ = @view u[odeform.dof_ranges[1]]
    @info t,norm(u)
    @unpack qrc, qrc_face, integrator = _integrator.cache.ltg_cache.inner_caches[1].K
    field_name = first(dh.field_names)
    u = zeros(getncells(dh.grid))
    for interface_cache in InterfaceIterator(dh)
        ip          = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
        interface_qr  = getquadraturerule(qrc_face, dh.subdofhandlers[1])
        interface_int_cache  = Thunderbolt.setup_interface_cache(integrator, interface_qr, ip, dh.subdofhandlers[1])
        Thunderbolt.estimate_kelly_interface!(Float64, u, (@view φ[Ferrite.interfacedofs(interface_cache)]), interface_cache, interface_int_cache)
    end
    u .= sqrt.(u)
    @info maximum(u)
    # sflat = ....?
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
        Thunderbolt.store_timestep_celldata!(file, t, u, "error ind")
        # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
        # for sidx in 1:Thunderbolt.num_states(ionic_model)
        #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
        # end
    end
end
TimerOutputs.print_timer()
# TimerOutputs.disable_debug_timings(Thunderbolt)
