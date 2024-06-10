abstract type AbstractLinearBlockAlgorithm <: LinearSolve.SciMLLinearSolveAlgorithm end # Why not the following? <: LinearSolve.AbstractLinearAlgorithm end

abstract type AbstractLinear2x2BlockAlgorithm <: AbstractLinearBlockAlgorithm end

@doc raw"""
    Schur2x2SaddleFormLinearSolver(inner_alg::AbstractLinearAlgorithm)

A solver for block systems of the form
```math
\begin{bmatrix}
    A_{11} & A_{12} \\
    A_{21} & 0
\end{bmatrix}
\begin{bmatrix}
    u_1 \\
    u_2
\end{bmatrix}
=
\begin{bmatrix}
    b_1 \\
    b_2
\end{bmatrix}
```
with small zero block of size $N_2 \times N_2$ and invertible $A_{11}$ with size $N_1 \times N_1$.
The inner linear solver is responsible to solve for $N_2$ systems of the form $A_{11} z_i = c_i$.
"""
struct Schur2x2SaddleFormLinearSolver{ASolverType <: LinearSolve.AbstractLinearAlgorithm} <: AbstractLinear2x2BlockAlgorithm
    inner_alg::ASolverType
end

# struct Schur2x2SaddleFormLinearSolverScratch{zType <: AbstractVector, wType <: AbstractMatrix}
#     z::zType
#     w::wType
# end

function allocate_scratch(alg::Schur2x2SaddleFormLinearSolver, A, b ,u)
    bs = blocksizes(A)
    return nothing
end

struct NestedLinearCache{AType, bType, innerSolveType, scratchType}
    A::AType
    b::bType
    innersolve::innerSolveType
    algscratch::scratchType
end

LinearSolve.default_alias_A(alg::AbstractLinearBlockAlgorithm, A::AbstractBlockMatrix, b) = true

function LinearSolve.init_cacheval(alg::AbstractLinear2x2BlockAlgorithm, A::AbstractBlockMatrix, b::AbstractVector, u::AbstractVector, Pl, Pr, maxiters::Int, abstol, reltol, verbose::Bool, assumptions::LinearSolve.OperatorAssumptions; zeroinit = true)
    # Check if input is okay
    bs = blocksizes(A)
    nblocks = length.(bs)
    @assert nblocks == (2,2) "Input matrix is not a 2x2 block matrix. Block sizes are actually $(nblocks)."
    @assert bs[1] == bs[2] "Diagonal blocks do not form quadratic matrices ($(bs[1]) != $(bs[2])). Aborting."

    # Transform Vectors to block vectors
    ub = PseudoBlockVector(u, bs[1])
    bb = PseudoBlockVector(b, bs[1])

    # Note that the inner solver does not solve for the u₁ via b₁, but for a helper zᵢ via cᵢ, which is then used to update u₁
    u₁ = @view ub[Block(1)]
    zᵢ = zero(u₁)
    b₁ = @view bb[Block(1)]
    cᵢ = zero(b₁)

    # Build inner helper
    A₁₁ = A[Block(1,1)]
    inner_prob = LinearSolve.LinearProblem(A₁₁, cᵢ; u0=zᵢ)
    innersolve = LinearSolve.init(
        inner_prob,
        alg.inner_alg;
        Pl, Pr,
        verbose,
        maxiters,
        reltol,
        abstol,
    )

    # Storage for intermediate values
    scratch = allocate_scratch(alg, A, b, u)
    return NestedLinearCache(A, b, innersolve, scratch)
end

function LinearSolve.solve!(cache::LinearSolve.LinearCache, alg::Schur2x2SaddleFormLinearSolver; kwargs...)
    # Instead of solving directly for
    #  / A₁₁  A₁₂ \ u₁ | b₁
    #  \ A₂₁  A₂₂ / u₂ | b₂
    # We rewrite the system as in Benzi, Golub, Liesen (2005) p.30
    # A₂₁ A₁₁⁻¹ A₁₂ u₂ = A₂₁ A₁₁⁻¹ b₁ - b₁
    #               u₁ = A₁₁⁻¹ b₁ - (A₁₁⁻¹ A₁₂ - A₂₂) u₂
    # which we rewrite as
    # A₂₁ z₂ u₂ = A₂₁ z₁ - b₂
    #        u₁ = z₁ - z₂ u₂
    # with
    #  A₁₁ z₁ = b₁
    #  A₁₁ z₂ = A₁₂
    # to avoid forming A₁₁⁻¹ explicitly.
    @unpack A,b = cache
    innercache = cache.cacheval
    @unpack algscratch, innersolve = innercache
    innersolve.isfresh = cache.isfresh

    bs = blocksizes(A)[1]

    u = zeros(sum(bs))# Where to query this?
    ub = PseudoBlockVector(u, bs)
    bb = PseudoBlockVector(b, bs)

    # First step is solving A₁₁ z₁ = b₁
    b₁ = @view bb[Block(1)]
    innersolve.b .= -b₁
    solz₁ = solve!(innersolve)
    z₁ = copy(solz₁.u) # TODO cache
    if !(LinearSolve.SciMLBase.successful_retcode(solz₁.retcode) || solz₁.retcode == LinearSolve.ReturnCode.Default)
        return LinearSolve.build_linear_solution(alg, u, nothing, cache; retcode = solz₁.retcode)
    end
    # Next step is solving for the transfer matrix A₁₁ z₂ = A₁₂
    z₂ = zeros(bs[1], bs[2]) # TODO cache
    A₁₂ = A[Block(1,2)]
    for i ∈ 1:bs[2]
        innersolve.b .= A₁₂[:,i]
        solz₂ = solve!(innersolve)
        z₂[:,i] .= solz₂.u
        if !(LinearSolve.SciMLBase.successful_retcode(solz₂.retcode) || solz₂.retcode == LinearSolve.ReturnCode.Default)
            return LinearSolve.build_linear_solution(alg, u, nothing, cache; retcode = solz₂.retcode)
        end
    end

    # Solve A₂₁ z₂ u₂ = A₂₁ z₁ - b₂
    A₂₁ = @view A[Block(2), Block(1)]
    A₂₂ = @view A[Block(2), Block(2)]
    u₂ = @view ub[Block(2)]
    b₂ = @view bb[Block(2)]
    A₂₁z₁ = A₂₁*z₁ # TODO cache
    A₂₁z₂C = A₂₁*z₂ - A₂₂ # TODO cache
    u₂ .= A₂₁z₂C \ -(b₂ + A₂₁z₁)

    # Solve for u₁ via u₁ = z₁ - z₂ u₂
    u₁ = @view ub[Block(1)]
    u₁ .= -(z₁+z₂*u₂)

    return LinearSolve.SciMLBase.build_linear_solution(alg, u, nothing, cache; retcode=LinearSolve.ReturnCode.Success)
end
