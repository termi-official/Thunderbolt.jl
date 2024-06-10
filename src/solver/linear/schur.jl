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

struct Schur2x2SaddleFormLinearSolverScratch{z1Type <: AbstractVector, z2Type <: AbstractMatrix, rhs2Type <: AbstractVector, sys2Type <: AbstractMatrix}
    z₁::z1Type
    z₂::z2Type
    A₂₁z₁₊b₂::rhs2Type
    A₂₁z₂₊A₂₂::sys2Type
end

function allocate_scratch(alg::Schur2x2SaddleFormLinearSolver, A, b ,u)
    bs = blocksizes(A)[1]
    return Schur2x2SaddleFormLinearSolverScratch(zeros(bs[1]), zeros(bs[1], bs[2]), zeros(bs[2]), zeros(bs[2], bs[2]))
end

struct NestedLinearCache{AType, bType, innerSolveType, scratchType}
    A::AType
    b::bType
    innersolve::innerSolveType
    algscratch::scratchType
end

# FIXME This does not work for some reason...
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
    A₁₁ = @view A[Block(1,1)]
    inner_prob = LinearSolve.LinearProblem(A₁₁, cᵢ; u0=zᵢ)
    @assert A₁₁ === inner_prob.A
    innersolve = LinearSolve.init(
        inner_prob,
        alg.inner_alg;
        alias_A = true,
        Pl, Pr,
        verbose,
        maxiters,
        reltol,
        abstol,
    )
    @assert A₁₁ === innersolve.A

    # Storage for intermediate values
    scratch = allocate_scratch(alg, A, b, u)
    return NestedLinearCache(A, b, innersolve, scratch)
end

function LinearSolve.solve!(cache::LinearSolve.LinearCache, alg::Schur2x2SaddleFormLinearSolver; kwargs...)
    # Instead of solving directly for
    #  / A₁₁  A₁₂ \ u₁ | b₁
    #  \ A₂₁  A₂₂ / u₂ | b₂
    # We rewrite the system as in Benzi, Golub, Liesen (2005) p.30
    # A₂₁ A₁₁⁻¹ A₁₂ u₂ = A₂₁ A₁₁⁻¹ b₁ - b₂
    #               u₁ = A₁₁⁻¹ b₁ - (A₁₁⁻¹ A₁₂ - A₂₂) u₂
    # which we rewrite as
    # A₂₁ z₂ u₂ = A₂₁ z₁ - b₂
    #        u₁ = z₁ - z₂ u₂
    # with the inner solves
    #  A₁₁ z₁ = b₁
    #  A₁₁ z₂ = A₁₂
    # to avoid forming A₁₁⁻¹ explicitly.
    @unpack A,b,u = cache
    innercache = cache.cacheval
    @unpack algscratch, innersolve = innercache

    # Unpack definitions into readable form without invoking copies
    @unpack z₁, z₂, A₂₁z₁₊b₂, A₂₁z₂₊A₂₂ = algscratch
    bs = blocksizes(A)[1]
    ub = PseudoBlockVector(u, bs)
    bb = PseudoBlockVector(b, bs)
    A₁₂ = @view A[Block(1, 2)]
    A₂₁ = @view A[Block(2, 1)]
    A₂₂ = @view A[Block(2, 2)]
    u₁ = @view ub[Block(1)]
    u₂ = @view ub[Block(2)]
    b₁ = @view bb[Block(1)]
    b₂ = @view bb[Block(2)]

    # Sync inner solver with outer solver
    innersolve.isfresh = cache.isfresh

    # First step is solving A₁₁ z₁ = b₁
    innersolve.b .= -b₁
    solz₁ = solve!(innersolve)
    z₁ .= solz₁.u
    if !(LinearSolve.SciMLBase.successful_retcode(solz₁.retcode) || solz₁.retcode == LinearSolve.ReturnCode.Default)
        @warn "A₁₁ z₁ = b₁ solve failed: $(solz₁.retcode)."
        return LinearSolve.SciMLBase.build_linear_solution(alg, u, solz₁.resid, cache; retcode = solz₁.retcode)
    end
    # Next step is solving for the transfer matrix A₁₁ z₂ = A₁₂
    for i ∈ 1:bs[2]
        innersolve.b .= A₁₂[:,i]
        solz₂ = solve!(innersolve)
        z₂[:,i] .= solz₂.u
        if !(LinearSolve.SciMLBase.successful_retcode(solz₂.retcode) || solz₂.retcode == LinearSolve.ReturnCode.Default)
            @warn "A₁₁ z₂ = A₁₂ ($i) solve failed"
            return LinearSolve.SciMLBase.build_linear_solution(alg, u, solz₂.resid, cache; retcode = solz₂.retcode)
        end
    end

    # Solve A₂₁ z₂ u₂ = A₂₁ z₁ - b₂
    A₂₁z₁₊b₂ = -(A₂₁*z₁ + b₂)
    mul!(A₂₁z₁₊b₂, A₂₁, z₁)
    A₂₁z₁₊b₂ .+= b₂
    A₂₁z₁₊b₂ .*= -1.0

    mul!(A₂₁z₂₊A₂₂, A₂₁, z₂)
    A₂₁z₂₊A₂₂ .-= A₂₂

    @info typeof(u₂)
    # ldiv!(u₂, A₂₁z₂₊A₂₂, A₂₁z₁₊b₂) # FIXME
    # ldiv!(A₂₁z₂₊A₂₂, A₂₁z₁₊b₂)
    u₂ .= A₂₁z₂₊A₂₂ \ A₂₁z₁₊b₂

    # Solve for u₁ via u₁ = z₁ - z₂ u₂
    mul!(u₁, z₂, u₂)
    u₁ .+= z₁
    u₁ .*= -1.0

    # Sync outer cache
    cache.isfresh = innersolve.isfresh
    return LinearSolve.SciMLBase.build_linear_solution(alg, u, nothing, cache; retcode=LinearSolve.ReturnCode.Success)
end

function inner_solve_schur(J::BlockMatrix, r::AbstractBlockArray)
    # TODO optimize
    Jdd = @view J[Block(1,1)]
    rd = @view r[Block(1)]
    rp = @view r[Block(2)]
    v = -(Jdd \ rd)
    Jdp = @view J[Block(1,2)]
    Jpd = @view J[Block(2,1)]
    w = Jdd \ Vector(Jdp[:,1])

    Jpdv = Jpd*v
    Jpdw = Jpd*w
    # Δp = [(rp[i] - Jpdv[i]) / Jpdw[i] for i ∈ 1:length(Jpdw)]
    Δp = (-rp[1] - Jpdv[1]) / Jpdw[1]
    wΔp = w*Δp
    Δd = -(v+wΔp) #-[-(v + wΔp[i]) for i in 1:length(v)]

    Δu = BlockVector([Δd; [Δp]], blocksizes(r,1))
    return Δu
end
