residual_norm(cache::AbstractNonlinearSolverCache, f::AbstractSemidiscreteFunction) = norm(cache.residual)
residual_norm(cache::AbstractNonlinearSolverCache, f::AbstractQuasiStaticFunction) = norm(cache.residual[Ferrite.free_dofs(getch(f))])
residual_norm(cache::AbstractNonlinearSolverCache, f::NullFunction, i::Block) = 0.0
residual_norm(cache::AbstractNonlinearSolverCache, f::NullFunction) = 0.0

eliminate_constraints_from_linearization!(cache::AbstractNonlinearSolverCache, f) = apply_zero!(cache.op.J, cache.residual, getch(f))
eliminate_constraints_from_increment!(Δu, f, cache::AbstractNonlinearSolverCache) = apply_zero!(Δu, getch(f))
function eliminate_constraints_from_increment!(Δu, f::AbstractSemidiscreteBlockedFunction, cache::AbstractNonlinearSolverCache)
    for (i,fi) ∈ enumerate(blocks(f))
        eliminate_constraints_from_increment!(Δu[Block(i)], fi, cache)
    end
end
eliminate_constraints_from_increment!(Δu, f::NullFunction, cache::AbstractNonlinearSolverCache) = nothing

function eliminate_constraints_from_linearization!(cache, f::AbstractSemidiscreteBlockedFunction)
    for (i,_) ∈ enumerate(blocks(f))
        eliminate_constraints_from_linearization_blocked!(cache, problem, Block(i))
    end
end

function eliminate_constraints_from_linearization_blocked!(cache, f::AbstractSemidiscreteBlockedFunction, i_::Block)
    @assert length(i_.n) == 1
    i = i_.n[1]
    fi = blocks(f)[i]
    hasproperty(fi, :ch) || return nothing
    ch = getch(fi)
    # TODO optimize this
    for j in 1:length( blocks(f))
        if i == j
            jacobian_block = getJ(cache.op, Block((i,i)))
            # Eliminate diagonal entry only
            residual_block = @view cache.residual[i_]
            apply_zero!(jacobian_block, residual_block, ch)
        else
            # Eliminate rows
            jacobian_block = getJ(cache.op, Block((i,j)))
            jacobian_block[ch.prescribed_dofs, :] .= 0.0
            # Eliminate columns
            jacobian_block = getJ(cache.op, Block((j,i)))
            jacobian_block[:, ch.prescribed_dofs] .= 0.0
        end
    end

    return nothing
end

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
