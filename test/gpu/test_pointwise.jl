# The reaction subproblem on its own. On a device solution vector the substepper's outer loop is a
# kernel launch (`ext/CuThunderboltExt.jl`), so this is what pins that launch: the same number of
# local systems, stepped by the same scheme, has to reach the same state as on the host.

@testset "Pointwise ODE solve" begin
    ode     = Thunderbolt.ParametrizedFHNModel{Float32}()
    npoints = 2^8
    nstates = num_states(ode)
    f       = Thunderbolt.PointwiseODEFunction(ode, nothing, 1:(nstates*npoints), :s)

    # A state per point that puts part of the population above the substepping threshold and part
    # below it, so both branches of the kernel are taken.
    u₀ = zeros(Float32, solution_size(f))
    u₀mat = reshape(u₀, (npoints, nstates))
    u₀mat[:, 1] .= range(-0.5f0, 1.5f0; length = npoints)
    u₀mat[:, 2] .= 0.1f0

    tspan = (0.0f0, 1.0f0)
    Δt = 0.01f0
    nsteps = 20

    solver(VT) =
        AdaptiveForwardEulerSubstepper(solution_vector_type = VT, reaction_threshold = 0.1f0)

    cpu =
        init(Thunderbolt.PointwiseODEProblem(f, copy(u₀), tspan), solver(Vector{Float32}); dt = Δt)
    gpu = init(
        Thunderbolt.PointwiseODEProblem(f, CuVector(u₀), tspan),
        solver(CuVector{Float32});
        dt = Δt,
    )

    for _ = 1:nsteps
        step!(cpu)
        step!(gpu)
    end

    @test gpu.u isa CuVector{Float32}
    @test Array(gpu.u) ≈ cpu.u
    # The step actually moved, so the comparison above is not two copies of the initial condition.
    @test cpu.u ≉ u₀
end
