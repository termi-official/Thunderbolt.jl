abstract type AbstractLinearBlockAlgorithm <: LinearSolve.AbstractLinearAlgorithm end

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
    @assert bs[1] = bs[2] "Diagonal blocks do not form quadratic matrices ($(bs[1]) != $(bs[2])). Aborting."

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
    inner_prob = LinearProblem(A₁₁, cᵢ; u0=zᵢ)
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

function LinearSolve.solve!(cache::NestedLinearCache, alg::Schur2x2SaddleFormLinearSolver; kwargs...)
    @unpack A,b,innersolve,scratch = cache
    innersolve.isfresh = cache.isfresh

    bs = blocksizes(b)[1]

    u = zeros(sum(bs))# Where to query this?
    ub = PseudoBlockVector(u, bs)

    # First step is solving A₁₁ z₁ = b₁
    b₁ = @view b[Block(1)]
    innersolve.b .= b₁
    solz₁ = solve!(innersolve)
    z₁ = copy(solz₁.u) # TODO cache
    if !(LinearSolve.LinearSolve.successful_retcode(solz₁.retcode) || solz₁.retcode == LinearSolve.ReturnCode.Default)
        return LinearSolve.build_linear_solution(alg, u, nothing, cache; retcode = solz₁.retcode)
    end
    # Next step is solving for the transfer matrix A₁₁ z₂ = A₁₂
    z₂ = zeros(bs[1], bs[2])
    for i ∈ 1:bs[2]
        innersolve.b .= b₁
        solz₂ = solve!(innersolve)
        z₂[:,i] .= solz₂.u
        if !(LinearSolve.LinearSolve.successful_retcode(solz₂.retcode) || solz₂.retcode == LinearSolve.ReturnCode.Default)
            return LinearSolve.build_linear_solution(alg, u, nothing, cache; retcode = solz₂.retcode)
        end
    end

    # Solve for u₂
    A₂₁z₁ = A₂₁*z₁ # TODO cache
    A₂₁z₂ = A₂₁*z₂ # TODO cache
    # TODO check that all are values in A₂₁z₂ are nonzeros or it is a solve failure
    u₂ = @view u[Block(2)]
    for i ∈ 1:bs[2]
        u₂[i] .= (-b₂[i] - A₂₁z₁[i]) / A₂₁z₂[i]
    end

    # Solve for u₁
    u₁ = @view u[Block(1)]
    u₁ .= -(z₁+z₂*u₂)

    return LinearSolve.build_linear_solution(alg, u, nothing, cache; retcode=LinearSolve.ReturnCode.Success)
end

# # TODO replace this with LinearSolve.jl
# # https://github.com/JuliaArrays/BlockArrays.jl/issues/319
# inner_solve(J, r) = J \ r
# function inner_solve_schur(J::BlockMatrix, r::BlockArray)
#     # TODO optimize
#     Jdd = @view J[Block(1,1)]
#     rd = @view r[Block(1)]
#     rp = @view r[Block(2)]
#     v = -(Jdd \ rd)
#     Jdp = @view J[Block(1,2)]
#     Jpd = @view J[Block(2,1)]
#     w = Jdd \ Vector(Jdp[:,1])

#     Jpdv = Jpd*v
#     Jpdw = Jpd*w
#     # Δp = [(rp[i] - Jpdv[i]) / Jpdw[i] for i ∈ 1:length(Jpdw)]
#     Δp = (-rp[1] - Jpdv[1]) / Jpdw[1]
#     wΔp = w*Δp
#     Δd = -(v+wΔp) #-[-(v + wΔp[i]) for i in 1:length(v)]

#     Δu = BlockVector([Δd; [Δp]], blocksizes(r,1))
#     return Δu
# end

# function inner_solve(J::BlockMatrix, r::BlockArray)
#     if length(blocksizes(r,1)) == 2 # TODO control by passing down the linear solver
#         return inner_solve_schur(J,r)
#     end

#     @timeit_debug "transform J " J_ = SparseMatrixCSC(J)
#     @timeit_debug "transform r" r_ = Vector(r)
#     @timeit_debug "direct solver" Δu = J_ \ r_
#     return Δu
# end
# inner_solve(J::BlockMatrix, r) = SparseMatrixCSC(J) \ r
# inner_solve(J, r::BlockArray) = J \ Vector(r)

# function solve_inner_linear_system!(Δu, cache::AbstractNonlinearSolverCache)
#     J = getJ(cache.op)
#     r = cache.residual
#     try
#         Δu .= inner_solve(J, r)
#     catch err
#         io = IOBuffer();
#         showerror(io, err, catch_backtrace())
#         error_msg = String(take!(io))
#         @warn "Linear solver failed: \n $error_msg"
#         return false
#     end
#     return true
# end
