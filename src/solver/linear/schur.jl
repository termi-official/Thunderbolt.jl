
# TODO replace this with LinearSolve.jl
# https://github.com/JuliaArrays/BlockArrays.jl/issues/319
inner_solve(J, r) = J \ r
function inner_solve_schur(J::BlockMatrix, r::BlockArray)
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

function inner_solve(J::BlockMatrix, r::BlockArray)
    if length(blocksizes(r,1)) == 2 # TODO control by passing down the linear solver
        return inner_solve_schur(J,r)
    end

    @timeit_debug "transform J " J_ = SparseMatrixCSC(J)
    @timeit_debug "transform r" r_ = Vector(r)
    @timeit_debug "direct solver" Δu = J_ \ r_
    return Δu
end
inner_solve(J::BlockMatrix, r) = SparseMatrixCSC(J) \ r
inner_solve(J, r::BlockArray) = J \ Vector(r)

function solve_inner_linear_system!(Δu, cache::AbstractNonlinearSolverCache)
    J = getJ(cache.op)
    r = cache.residual
    try
        Δu .= inner_solve(J, r)
    catch err
        io = IOBuffer();
        showerror(io, err, catch_backtrace())
        error_msg = String(take!(io))
        @warn "Linear solver failed: \n $error_msg"
        return false
    end
    return true
end
