using Thunderbolt, Thunderbolt.TimerOutputs
using UnPack # TODO REMOVEME
import Thunderbolt.OS
######################################################

function spiral_wave_initializer!(u₀, problem::Thunderbolt.SplitProblem, t₀)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    dh = problem.A.dh
    ionic_model = problem.B.ode
    # TODO extraction these via index maps
    φ₀ = @view u₀[1:ndofs(dh)];
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)));
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        φₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        # TODO query coordinate directly from the cell model
        coordinates = getcoordinates(cell)
        for (i, (x₁, x₂)) in zip(φₘ_celldofs,coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                φ₀[i] = 1.0
            end
            if x₂ >= 1.25
                s₀[i,1] = 0.1
            end
        end
    end
end

tspan = (0.0, 1000.0)
dtvis = 25.0
dt₀ = 1.0

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5))),
    NoStimulationProtocol(),
    Thunderbolt.FHNModel()
)

mesh = generate_mesh(Quadrilateral, (2^7, 2^7), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

odeform = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)
# TODO this should be done by the function above.
ndofsφ = ndofs(odeform.A.dh)
nstates_per_cell = Thunderbolt.num_states(odeform.B.ode)
splitfun = OS.GenericSplitFunction(
    (odeform.A, odeform.B),
    (1:ndofsφ, 1:(1+nstates_per_cell)*ndofsφ)
)

u₀ = zeros(Float64, OS.function_size(splitfun))
spiral_wave_initializer!(u₀, odeform, 0.0)

problem = OS.OperatorSplittingProblem(splitfun, u₀, tspan)

timestepper = OS.LieTrotterGodunov((
    BackwardEulerSolver(),
    ForwardEulerCellSolver(),
))

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

io = ParaViewWriter("spiral-wave-test")

TimerOutputs.enable_debug_timings(Thunderbolt)
# TimerOutputs.enable_debug_timings(Main)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    dh = odeform.A.dh
    φ = u[1:ndofs(dh)]
    @info t,norm(u)
    # φ = @view u[odeform.subproblems[1].indexset]
    # sflat = ....?
    store_timestep!(io, t, dh.grid)
    Thunderbolt.store_timestep_field!(io, t, dh, φ, :φₘ) # TODO allow views
    # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
    # for sidx in 1:Thunderbolt.num_states(ionic_model)
    #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
    # end
    Thunderbolt.finalize_timestep!(io, t)
end
TimerOutputs.print_timer()
# TimerOutputs.disable_debug_timings(Main)
TimerOutputs.disable_debug_timings(Thunderbolt)

# v2
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Time                    Allocations
# ───────────────────────   ────────────────────────
# Tot / % measured:                           3.66s /  88.2%            138MiB /  17.2%

# Section                                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Subintegrators                                        1.00k    3.23s  100.0%  3.23ms   23.7MiB  100.0%  24.3KiB
# inner solve                                         1.00k    2.70s   83.7%  2.70ms    130KiB    0.5%     133B
# cell loop                                           1.00k    184ms    5.7%   184μs     0.00B    0.0%    0.00B
# b = M uₙ₋₁                                          1.00k    162ms    5.0%   162μs   46.9KiB    0.2%    48.0B
# implicit_euler_heat_solver_update_system_matrix!        1   1.31ms    0.0%  1.31ms   7.04MiB   29.7%  7.04MiB
# update source term                                  1.00k    223μs    0.0%   223ns     0.00B    0.0%    0.00B
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# v3
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Time                    Allocations
# ───────────────────────   ────────────────────────
# Tot / % measured:                          4.31s /  81.7%            142MiB /   5.0%

# Section                                            ncalls     time    %tot     avg     alloc    %tot      avg
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# inner solve                                         1.00k    3.11s   88.5%  3.11ms    130KiB    1.8%     133B
# cell loop                                           1.00k    199ms    5.7%   199μs     0.00B    0.0%    0.00B
# b = M uₙ₋₁                                          1.00k    178ms    5.1%   178μs     0.00B    0.0%    0.00B
# implicit_euler_heat_solver_update_system_matrix!        1   25.1ms    0.7%  25.1ms   7.04MiB   98.2%  7.04MiB
# update source term                                  1.00k    247μs    0.0%   247ns     0.00B    0.0%    0.00B
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
