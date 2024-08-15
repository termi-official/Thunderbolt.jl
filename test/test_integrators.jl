import Thunderbolt: OS, ThunderboltTimeIntegrator
# using BenchmarkTools
using UnPack

@testset "Operator Splitting API" begin

ODEFunction = Thunderbolt.DiffEqBase.ODEFunction

# For testing purposes
struct DummyForwardEuler
end

mutable struct DummyForwardEulerCache{duType, uType} <: Thunderbolt.AbstractTimeSolverCache
    du::duType
    uₙ::uType
    uₙ₋₁::uType
end

# Dispatch for leaf construction
function OS.construct_inner_cache(f::ODEFunction, alg::DummyForwardEuler, u::AbstractArray, uprev::AbstractArray)
    DummyForwardEulerCache(copy(uprev), copy(uprev), copy(uprev))
end

# Dispatch innermost solve
function OS.advance_solution_to!(integ::ThunderboltTimeIntegrator, cache::DummyForwardEulerCache, tend)
    @unpack f, dt, u, p, t = integ
    @unpack du = cache

    f(du, u, p, t)
    @. u += dt * du
end

# Operator splitting

# Reference
function ode_true(du, u, p, t)
    du .= -0.1u
    du[1] += 0.01u[3]
    du[3] += 0.01u[1]
end

# Setup individual functions
# Diagonal components
function ode1(du, u, p, t)
    @. du = -0.1u
end
# Offdiagonal components
function ode2(du, u, p, t)
    du[1] = 0.01u[2]
    du[2] = 0.01u[1]
end

f1 = ODEFunction(ode1)
f2 = ODEFunction(ode2)

# Here we describe index sets f1dofs and f2dofs that map the
# local indices in f1 and f2 into the global problem. Just put
# ode_true and ode1/ode2 side by side to see how they connect.
f1dofs = [1,2,3]
f2dofs = [1,3]
fsplit1 = GenericSplitFunction((f1,f2), (f1dofs, f2dofs))

# Now the usual setup just with our new problem type.
# u0 = rand(3)
u0 = [0.7611944793397108
      0.9059606424982555
      0.5755174199139956]
tspan = (0.0,100.0)
prob = OperatorSplittingProblem(fsplit1, u0, tspan)

# Now some recursive splitting
function ode3(du, u, p, t)
    du[1] = 0.005u[2]
    du[2] = 0.005u[1]
end
f3 = ODEFunction(ode3)
# The time stepper carries the individual solver information.

@testset "OperatorSplitting" begin
    

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    f3dofs = [1,3]
    fsplit2_inner = GenericSplitFunction((f3,f3), (f3dofs, f3dofs))
    fsplit2_outer = GenericSplitFunction((f1,fsplit2_inner), (f1dofs, f2dofs))
    for TimeStepperType in (LieTrotterGodunov,)
        for controller in (Thunderbolt.ReactionTangentController(0.5, 1.0, (0.01, 0.3)),) 
            timestepper = TimeStepperType(
                (DummyForwardEuler(), DummyForwardEuler())
            )
            timestepper_adaptive = Thunderbolt.AdaptiveOperatorSplittingAlgorithm(timestepper, controller)
            timestepper_inner = TimeStepperType(
                (DummyForwardEuler(), DummyForwardEuler())
            )
            timestepper_inner_adaptive = Thunderbolt.AdaptiveOperatorSplittingAlgorithm(timestepper_inner, controller) #TODO: Copy the controller instead
            timestepper2 = TimeStepperType(
                (DummyForwardEuler(), timestepper_inner)
            )
            timestepper2_adaptive = Thunderbolt.AdaptiveOperatorSplittingAlgorithm(timestepper2, controller)

            for (tstepper1, tstepper_inner, tstepper2) in (
                    (timestepper, timestepper_inner, timestepper2),
                    # (timestepper_adaptive, timestepper_inner_adaptive, timestepper2_adaptive)
                    )
                # The remaining code works as usual.
                integrator = DiffEqBase.init(prob, tstepper1, dt=0.01, verbose=true)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal = copy(integrator.u)
                @test ufinal ≉ u0 # Make sure the solve did something

                DiffEqBase.reinit!(integrator, u0; tspan)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0)
                end
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                @test  isapprox(ufinal, integrator.u, atol=1e-8)

                DiffEqBase.reinit!(integrator, u0; tspan)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
                end
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                @test  isapprox(ufinal, integrator.u, atol=1e-8)

                DiffEqBase.reinit!(integrator, u0; tspan)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success

                prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)
                integrator2 = DiffEqBase.init(prob2, tstepper2, dt=0.01, verbose=true)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success

                DiffEqBase.reinit!(integrator2, u0; tspan)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (u, t) in DiffEqBase.TimeChoiceIterator(integrator2, 0.0:5.0:100.0)
                end
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                @test isapprox(ufinal, integrator2.u, atol=1e-8)

                DiffEqBase.reinit!(integrator2, u0; tspan)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Success
            end
            # integrator = DiffEqBase.init(prob, timestepper, dt=0.01, verbose=true)
            # for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0) end
            # integrator_adaptive = DiffEqBase.init(prob, timestepper_adaptive, dt=0.01, verbose=true)
            # for (u, t) in DiffEqBase.TimeChoiceIterator(integrator_adaptive, 0.0:5.0:100.0) end
            # @test  isapprox(integrator_adaptive.u, integrator.u, atol=1e-5)
        end
    end
end

# tnext = tspan[1]+0.01
# @btime OS.advance_solution_to!($integrator, $tnext) setup=(DiffEqBase.reinit!(integrator, u0; tspan))
#   326.743 ns (8 allocations: 416 bytes) for 1 (OUTDATED
#   89.949 ns (0 allocations: 0 bytes) for 2 (OUTDATED
#   31.418 ns (0 allocations: 0 bytes) for 3
# @btime DiffEqBase.solve!($integrator) setup=(DiffEqBase.reinit!(integrator, u0; tspan));
#   431.632 μs (10000 allocations: 507.81 KiB) for 1 (OUTDATED
#   105.712 μs (0 allocations: 0 bytes) for 2 (OUTDATED)
#   1.852 μs (0 allocations: 0 bytes) for 3

end
