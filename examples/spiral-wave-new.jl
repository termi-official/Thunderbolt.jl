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
dtvis = 10.0
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
splitfun = OS.GenericSplitFunction(
    (odeform.A, odeform.B),
    [1:ndofs(odeform.A.dh), 1:2*ndofs(odeform.A.dh)]
)

# TODO query with function from odeform
u₀ = zeros(ndofs(odeform.A.dh) + ndofs(odeform.A.dh)*Thunderbolt.num_states(odeform.B.ode))
spiral_wave_initializer!(u₀, odeform, 0.0)

# FIXME
OS.recursive_null_parameters(stuff) = OS.DiffEqBase.NullParameters()
problem = OS.OperatorSplittingProblem(splitfun, u₀, tspan)

# TODO put this into Thunderbolt. This is essentially perform_step!
function OS.step_inner!(integ, cache::Thunderbolt.BackwardEulerSolverCache)
    @unpack f, dt, u, uprev, p, t = integ
    # TODO move Δt_last out of the cache into a preamble step
    # TODO remove uₙ, uₙ₋₁ from cache
    @unpack Δt_last, b, M, A, #=uₙ, uₙ₋₁,=# linsolver = cache

    # Prepare right hand side b = M uₙ₋₁
    @timeit_debug "b = M uₙ₋₁" Thunderbolt.mul!(b, M, uprev)
    # TODO replace this section with an AffineOperator
    # Update source term
    @timeit_debug "update source term" begin
        Thunderbolt.implicit_euler_heat_update_source_term!(cache, t)
        add!(b, cache.source_term)
    end
    # TODO use an affine operator
    dt ≈ Δt_last || Thunderbolt.implicit_euler_heat_solver_update_system_matrix!(cache, dt)
    # Solve linear problem
    # TODO abstraction layer and way to pass the solver/preconditioner pair (LinearSolve.jl?)
    @timeit_debug "inner solve" Thunderbolt.Krylov.cg!(linsolver, A, b, uprev)
    @inbounds u .= linsolver.x
    @info linsolver.stats
    return nothing
end

# function perform_step!(cell_model::ION, t::Float64, Δt::Float64, solver_cache::ForwardEulerCellSolverCache{VT}) where {VT, ION <: AbstractIonicModel}
function OS.step_inner!(integ, cache::Thunderbolt.ForwardEulerCellSolverCache)
    @unpack f, dt, u, uprev, p, t = integ
    # Remove these from the cache
    @unpack du#=, uₙ, sₙ=# = cache
    # TODO eliminate this
    midpoint = length(u) ÷ 2
    uₙ = @view u[1:midpoint]
    sₙ = @view u[(midpoint+1):end]

    # TODO formulate as a kernel for GPU
    @timeit_debug "cell loop" for i ∈ 1:length(uₙ)
        @inbounds φₘ_cell = uₙ[i]
        @inbounds s_cell  = @view sₙ[i,:]

        # Should be something like a DiffEqBase.SplitFunction
        # instance_parameters = nothing # TODO fill with x and Cₘ
        # f(du, φₘ_cell, s_cell, t, instance_parameters)
        Thunderbolt.cell_rhs!(du, φₘ_cell, s_cell, nothing, t, f.ode)

        # TODO subindices via f
        @inbounds uₙ[i] = φₘ_cell + dt*du[1]

        # Non-allocating assignment
        @inbounds for j ∈ 1:Thunderbolt.num_states(f.ode)
            sₙ[i,j] = s_cell[j] + dt*du[j+1]
        end
    end

    return nothing
end

# REMOVEME
# v2
function OS.build_subintegrators_recursive(f::Thunderbolt.TransientHeatProblem, p::Any, cache::Any, u::SubArray, uprev::SubArray, t, dt)
    return OS.ThunderboltSubIntegrator(f, u, uprev, p, t, dt)
end
function OS.build_subintegrators_recursive(f::Thunderbolt.AbstractPointwiseProblem, p::Any, cache::Any, u::SubArray, uprev::SubArray, t, dt)
    return OS.ThunderboltSubIntegrator(f, u, uprev, p, t, dt)
end
# v3
function OS.build_subintegrators_recursive(f::Thunderbolt.TransientHeatProblem, p::Any, cache::Thunderbolt.BackwardEulerSolverCache, u::AbstractArray, uprev::AbstractArray, t, dt, dof_range, umaster)
    return OS.ThunderboltSubIntegrator(f, u, umaster, uprev, dof_range, p, t, dt)
end
function OS.build_subintegrators_recursive(f::Thunderbolt.AbstractPointwiseProblem, p::Any, cache::Thunderbolt.ForwardEulerCellSolverCache, u::AbstractArray, uprev::AbstractArray, t, dt, dof_range, umaster)
    return OS.ThunderboltSubIntegrator(f, u, umaster, uprev, dof_range, p, t, dt)
end

timestepper = OS.LieTrotterGodunov((
    BackwardEulerSolver(),
    ForwardEulerCellSolver(),
))

# Dispatch for leaf construction
function OS.construct_inner_cache(f, alg::ForwardEulerCellSolver, u::AbstractArray, uprev::AbstractArray)
    Thunderbolt.setup_solver_cache(f, alg, 0.0)
end
function OS.construct_inner_cache(f, alg::BackwardEulerSolver, u::AbstractArray, uprev::AbstractArray)
    return Thunderbolt.setup_solver_cache(f, alg, 0.0)
end

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

io = ParaViewWriter("test_new")

state_symbol(ionic_model, sidx) = Symbol("s$sidx") # TODO query via ionic model and into Thunderbolt.jl

TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.enable_debug_timings(Main)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    # # @show t
    # dh = odeform.A.dh
    # φ = u[1:ndofs(dh)]
    # # φ = @view u[odeform.subproblems[1].indexset]
    # # sflat = ....?
    # store_timestep!(io, t, dh.grid)
    # Thunderbolt.store_timestep_field!(io, t, dh, φ, :φₘ) # TODO allow views
    # # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
    # # for sidx in 1:Thunderbolt.num_states(ionic_model)
    # #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
    # # end
    # Thunderbolt.finalize_timestep!(io, t)
end
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Main)
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
