using Thunderbolt

######################################################
mutable struct IOCallback{IO}
    io::IO
    counter::Int
end

IOCallback(io) = IOCallback(io, -1)

function (iocb::IOCallback{ParaViewWriter{PVD}})(t, problem::Thunderbolt.SplitProblem, solver_cache) where {PVD}
    # iocb.counter += 1
    # (iocb.counter % 10) == 0 || return nothing
    # store_timestep!(iocb.io, t, problem.A.dh.grid)
    # Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.A_solver_cache.uₙ, :φₘ)
    # Thunderbolt.store_timestep_field!(iocb.io, t, problem.A.dh, solver_cache.B_solver_cache.sₙ[1:length(solver_cache.A_solver_cache.uₙ)], :s)
    # Thunderbolt.finalize_timestep!(iocb.io, t)
    return nothing
end

######################################################
function spiral_wave_initializer(problem::Thunderbolt.SplitProblem, t₀)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    dh = problem.A.dh
    ionic_model = problem.B.ode
    u₀ = zeros(ndofs(dh));
    s₀ = zeros(ndofs(dh), Thunderbolt.num_states(ionic_model));
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        # TODO get coordinate via coordinate_system
        coordinates = getcoordinates(cell)
        for (i, (x₁, x₂)) in enumerate(coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                u₀[ϕₘ_celldofs[i]] = 1.0
            end
            if x₂ >= 1.25
                s₀[ϕₘ_celldofs[i],1] = 0.1
            end
        end
    end
    return u₀, s₀
end

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5))),
    NoStimulationProtocol(),
    Thunderbolt.FHNModel()
)

mesh = generate_mesh(Quadrilateral, (2^7, 2^7), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)

solver = LTGOSSolver(
    BackwardEulerSolver(),
    ForwardEulerCellSolver()
)

# Idea: We want to solve a semidiscrete problem, with a given compatible solver, on a time interval, with a given initial condition
# TODO iterator syntax
using Thunderbolt.TimerOutputs
TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
solve(
    problem,
    solver,
    1.0,
    (0.0, 1000.0),
    spiral_wave_initializer,
    IOCallback(ParaViewWriter("test"))
)

TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)

# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Time                    Allocations
# ───────────────────────   ────────────────────────
# Tot / % measured:                           4.49s /  85.4%           89.3MiB /  39.9%

# Section                                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# step A                                                1.00k    3.33s   86.8%  3.33ms   28.4MiB   79.7%  29.1KiB
# inner solve                                         1.00k    2.99s   78.0%  2.99ms    130KiB    0.4%     133B
# b = M uₙ₋₁                                          1.00k    182ms    4.7%   182μs     0.00B    0.0%    0.00B
# implicit_euler_heat_solver_update_system_matrix!        2   8.77ms    0.2%  4.39ms   11.8MiB   33.1%  5.91MiB
# update source term                                  1.00k    209μs    0.0%   209ns     0.00B    0.0%    0.00B
# initial assembly                                          1    302ms    7.9%   302ms   7.26MiB   20.3%  7.26MiB
# assemble element                                    32.8k   11.0ms    0.3%   334ns     0.00B    0.0%    0.00B
# step B                                                1.00k    205ms    5.3%   204μs     0.00B    0.0%    0.00B
# transfer A->B                                         1.00k    259μs    0.0%   259ns     0.00B    0.0%    0.00B
# transfer B->A                                         1.00k    242μs    0.0%   242ns     0.00B    0.0%    0.00B
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
