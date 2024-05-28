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
    du .= -0.1u
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
f = GenericSplitFunction((f1,f2), [f1dofs, f2dofs])

# Now the usual setup just with our new problem type.
u0 = rand(3)
tspan = (0.0,100.0)
prob = OperatorSplittingProblem(f, u0, tspan)

# The time stepper carries the individual solver information.
timestepper = LieTrotterGodunov(
    (ForwardEuler(), ForwardEuler())
)

# The remaining code works as usual.
integrator = DiffEqBase.init(prob, timestepper, dt=0.01, verbose=true)
DiffEqBase.solve!(integrator)
ufinal = copy(integrator.u)

DiffEqBase.reinit!(integrator, u0; t0=tspan[1], tf=tspan[2])
for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0)
    @show u, t
end
@assert ufinal == integrator.u

DiffEqBase.reinit!(integrator, u0; t0=tspan[1], tf=tspan[2])
for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
    @show tprev, t
end
@assert ufinal == integrator.u
