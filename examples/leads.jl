using Thunderbolt, StaticArrays, DelimitedFiles

# ######################################################
mutable struct IOCallback{IO, LF, ECGRC, VT, κT}
    io::IO
    counter::Int
    lead_field::LF
    ecg_reconst_cache::ECGRC
    v::VT
    κᵢ::κT
end

IOCallback(io, lead_field, ecg_reconst_cache, v, κᵢ) = IOCallback(io, -1, lead_field, ecg_reconst_cache, v, κᵢ)
function (iocb::IOCallback{ParaViewWriter{PVD}})(t, problem::Thunderbolt.SplitProblem, solver_cache) where {PVD}
    iocb.counter += 1
    (iocb.counter % 100) == 0 || return nothing
    ϕₘ = solver_cache.A_solver_cache.uₙ
    Thunderbolt.reinit!(iocb.ecg_reconst_cache, ϕₘ, iocb.κᵢ)
    Thunderbolt.reinit!(iocb.lead_field, ϕₘ, iocb.κᵢ)
    φₑ = iocb.ecg_reconst_cache.ϕₑ
    store_timestep!(iocb.io, t, problem.A.dh.grid)
    push!(iocb.v, Thunderbolt.evaluate_ecg(iocb.lead_field, (Vec(20., 0., 0.), Vec(0., 0., 0.)))) 
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, φₑ, :φₑ)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, ϕₘ, :φₘ)
    Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.B_solver_cache.sₙ[1:length(solver_cache.A_solver_cache.uₙ)], :s)
    Thunderbolt.finalize_timestep!(iocb.io, t)
    return nothing
end

function steady_state_initializer(problem::Thunderbolt.SplitProblem, t₀)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    dh = problem.A.dh
    ionic_model = problem.B.ode
    default_values = Thunderbolt.default_initial_state(ionic_model)
    u₀ = ones(ndofs(dh))*default_values[1]
    s₀ = zeros(ndofs(dh), Thunderbolt.num_states(ionic_model));
    for i ∈ 1:Thunderbolt.num_states(ionic_model)
        s₀[:, i] .= default_values[1+i]
    end
    return u₀, s₀
end

######################################################

mesh = generate_grid(Hexahedron, (80,28,12), Vec((0.0,0.0,0.0)), Vec((20.0,7.0,3.0)))
cs = CartesianCoordinateSystem(mesh)

qr_collection = QuadratureRuleCollection(2)
ip_collection = LagrangeCollection{1}()

ms = OrthotropicMicrostructureModel(
    ConstantCoefficient(Vec(0., 0., 1.)),
    ConstantCoefficient(Vec(1., 0., 0.)),
    ConstantCoefficient(Vec(0., 1., 0.))
)

κ = SpectralTensorCoefficient(
    ms, 
    ConstantCoefficient(SVector(0.3, 0.12, 0.12) + SVector(0.3, 0.03, 0.03)))
κᵢ = SpectralTensorCoefficient(
    ms, 
    ConstantCoefficient(SVector(0.3, 0.12, 0.12)))


vtk_grid("Lead", lead_field.op.dh) do vtk
    vtk_point_data(vtk, lead_field.op.dh, lead_field.Z[1,:])
end

model = MonodomainModel(
    ConstantCoefficient(1),
    ConstantCoefficient(1),
    κᵢ,
    Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
        AnalyticalCoefficient((x,t) -> norm(x) < 1.5 && t < 2.0 ? 0.5 : 0.0, Thunderbolt.CoordinateSystemCoefficient(cs)),
        [SVector((0.0, 2.1))]
    ),
    Thunderbolt.PCG2019()
)

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => ip_collection)),
    mesh
)

lead_field = Thunderbolt.Geselowitz1989ECGLeadCache(problem, κ, [Vec(20., 0., 0.), Vec(0., 0., 0.)], [(1,2)])

ecg_reconst_cache = Thunderbolt.ECGPoissonReconstructionCache(problem, κ)


solver = LTGOSSolver(
    BackwardEulerSolver(),
    ForwardEulerCellSolver()
)

iocb_ = IOCallback(ParaViewWriter("Niederer"), lead_field, ecg_reconst_cache, Float64[], κᵢ)

# Idea: We want to solve a semidiscrete problem, with a given compatible solver, on a time interval, with a given initial condition
# TODO iterator syntax
solve(
    problem,
    solver,
    0.01,
    (0.0, 50.0),
    steady_state_initializer,
    iocb_
)