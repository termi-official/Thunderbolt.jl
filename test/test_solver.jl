using Test, Thunderbolt
import LinearSolve: KrylovJL_GMRES, LinearProblem
using BlockArrays

@testset "Linear Solver" begin
    alg = Thunderbolt.SchurComplementLinearSolver(KrylovJL_GMRES())

    A = BlockArray{Float64}(undef_blocks, [1, 1], [1, 1])
    A[Block(1), Block(1)] = ones(1, 1)
    A[Block(1), Block(2)] = ones(1, 1)
    A[Block(2), Block(1)] = ones(1, 1)
    A[Block(2), Block(2)] = zeros(1, 1)
    b = rand(2)
    u0 = zeros(2)

    prob = LinearProblem(A, b; u0)
    linsolve = init(prob, alg)
    sol = solve!(linsolve)

    @test A\b ≈ sol.u

    s1 = 5
    s2 = 3
    A = BlockArray{Float64}(undef_blocks, [s1, s2], [s1, s2])
    A[Block(1), Block(1)] = rand(s1, s1)
    A[Block(1), Block(2)] = rand(s1, s2)
    A[Block(2), Block(1)] = rand(s2, s1)
    A[Block(2), Block(2)] = rand(s2, s2)
    b = rand(s1+s2)
    u0 = zeros(s1+s2)

    prob = LinearProblem(A, b; u0)
    linsolve = init(prob, alg)
    sol = solve!(linsolve)

    @test A\b ≈ sol.u
end

@testset "NullFunction stage operator" begin
    # Pins the NullOperator import: both setup_stage_operator methods answering a
    # NullFunction construct one, and neither path is reached anywhere else in the suite.
    f = Thunderbolt.NullFunction(3)
    solver = HomotopyPathSolver(NewtonRaphsonSolver(; max_iter = 1))
    op = Thunderbolt.setup_stage_operator(f, solver, nothing, 0.0)
    @test op isa Thunderbolt.NullOperator
    @test size(op) == (3, 3)
    op2 = Thunderbolt.setup_stage_operator(f, BackwardEulerSolver(), nothing, 0.0)
    @test op2 isa Thunderbolt.NullOperator
end
