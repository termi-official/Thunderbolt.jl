using Thunderbolt, CUDA

import Thunderbolt: solution_size, num_states, FHNModel

using Adapt

function uniform_initializer!(u₀, f::PointwiseODEFunction)
    ionic_model = f.ode
    ndofs       = solution_size(f)
    nstates     = num_states(ionic_model)
    u₀mat       = reshape(u₀, (ndofs ÷ nstates, nstates))
    φ₀          = @view u₀mat[:, 1];
    s₀          = @view u₀mat[:, 2:end];

    for i in 1:f.npoints
        φ₀[i]   = 1.0
        s₀[i,1] = 0.1
    end
end

odefun = PointwiseODEFunction(
    2,
    Thunderbolt.ParametrizedFHNModel{Float32}(),
)

u₀ = zeros(Float32, solution_size(odefun))
uniform_initializer!(u₀, odefun)

tspan = (0.0, 10.0)
dt₀   = 0.1
dtvis = 10.0

cputimestepper = ForwardEulerCellSolver(solution_vector_type=Vector{Float32})
cpuproblem     = PointwiseODEProblem(odefun, u₀, tspan)
cpuintegrator  = init(cpuproblem, cputimestepper, dt=dt₀)

for (u, t) in TimeChoiceIterator(cpuintegrator, tspan[1]:dtvis:tspan[2])
    @info (u, t)
end

uniform_initializer!(u₀, odefun)
u₀gpu          = CuVector(u₀)
gputimestepper = ForwardEulerCellSolver(solution_vector_type=CuVector{Float32}, batch_size_hint=1)
gpuproblem     = PointwiseODEProblem(odefun, u₀gpu, tspan)
gpuintegrator  = init(gpuproblem, gputimestepper, dt=dt₀)

for (u, t) in TimeChoiceIterator(gpuintegrator, tspan[1]:dtvis:tspan[2])
    @info (u, t)
end

try 
    Thunderbolt._pointwise_step_outer_kernel!(odefun,1.0, 1.0, gpuintegrator.cache, u₀gpu)
catch err
    code_typed(err; interactive = true)
end
