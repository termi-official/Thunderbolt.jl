# Operator splitting
ode1(du, u, p, t) = du .= -0.1u

function ode2(du, u, p, t)
    du[1] = 0.01u[2]
    du[2] = 0.01u[1]
end

f1 = ODEFunction(ode1)
f2 = ODEFunction(ode2)

# partitioning = Dict(:u₁ => 1:1:3, :u₂ => 1:2:3)

# TODO ctor
# f = GenericSplitFunction(partitioning, (f1, [:u₁, :u₂]), (f2, [:u₂]))
# f = GenericSplitFunction((f1,f2), [[:u₁, :u₂], [:u₂]], partitioning)
f = GenericSplitFunction((f1,f2), [[1,2,3], [1,3]])
u0 = rand(3)
tspan = (0.0,100.0)
prob = OperatorSplittingProblem(f, u0, tspan)

timestepper = LieTrotterGodunov(
    (ForwardEuler(), ForwardEuler())
)

integrator = DiffEqBase.init(prob, timestepper, dt=0.01, verbose=true)

# DiffEqBase.solve!(integrator)

for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0)
    @show u, t
end

for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
    @show tprev, t
end

# Reference
function ode_true(du, u, p, t)
    du .= -0.1u
    du[1] += 0.01u[3]
    du[3] += 0.01u[1]
end
