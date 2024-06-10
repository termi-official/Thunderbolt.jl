using Thunderbolt, LinearSolve, BlockArrays

# A = BlockArray{Float64}(undef_blocks, [1,1], [1,1])
# A[Block(1), Block(1)] = ones(1,1)
# A[Block(1), Block(2)] = ones(1,1)
# A[Block(2), Block(1)] = ones(1,1)
# A[Block(2), Block(2)] = zeros(1,1)

s1 = 5
s2 = 2
A = BlockArray{Float64}(undef_blocks, [s1,s2], [s1,s2])
A[Block(1), Block(1)] = rand(s1,s1)
A[Block(1), Block(2)] = rand(s1,s2)
A[Block(2), Block(1)] = rand(s2,s1)
A[Block(2), Block(2)] = zeros(s2,s2)
b = rand(s1+s2)
u0 = zeros(s1+s2)

alg = Thunderbolt.Schur2x2SaddleFormLinearSolver(KrylovJL_GMRES())
prob = LinearProblem(A, b; u0)
linsolve = init(prob, alg)
sol = solve!(linsolve)

@assert A\b ≈ sol.u
