"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
end

mutable struct NewtonRaphsonSolverCache{OpType, ResidualType, T} <: AbstractNonlinearSolverCache
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonRaphsonSolver{T}
    #linear_solver_cache
end

function setup_solver_cache(problem, solver::NewtonRaphsonSolver{T}) where {T}
    NewtonRaphsonSolverCache(setup_operator(problem, solver), Vector{T}(undef, solution_size(problem)), solver)
end

function setup_solver_cache(problem::AbstractCoupledProblem, solver::NewtonRaphsonSolver{T}) where {T}
    residual_buffer = mortar([
        Vector{T}(undef, solution_size(base_problem)) for base_problem ∈ base_problems(problem)
    ])
    NewtonRaphsonSolverCache(setup_operator(problem, solver), residual_buffer, solver)
end

function setup_solver_cache(coupled_problem::CoupledProblem, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack base_problems = coupled_problem
    op = BlockOperator((
        [i == j ? setup_operator(coupled_problem, i, solver) : setup_coupling_operator(coupled_problem, i, j, solver) for i in 1:length(base_problems) for j in 1:length(base_problems)]...,
    ))
    solution = mortar([
        Vector{T}(undef, solution_size(base_problems[i])) for i ∈ 1:length(base_problems)
    ])

    NewtonRaphsonSolverCache(op, solution, solver)
end

residual_norm(solver_cache::NewtonRaphsonSolverCache, problem) = norm(solver_cache.residual[Ferrite.free_dofs(getch(problem))])
residual_norm(solver_cache::NewtonRaphsonSolverCache, problem, i::Block) = norm(solver_cache.residual[i][Ferrite.free_dofs(getch(problem))])
residual_norm(solver_cache::NewtonRaphsonSolverCache, problem::NullProblem, i::Block) = 0.0

###########################################################################################################
eliminate_constraints_from_linearization!(solver_cache, problem) = apply_zero!(solver_cache.op.J, solver_cache.residual, getch(problem))
eliminate_constraints_from_increment!(Δu, problem, solver_cache) = apply_zero!(Δu, getch(problem))

function eliminate_constraints_from_increment!(Δu, problem::AbstractCoupledProblem, solver_cache)
    for (i,p) ∈ enumerate(base_problems(problem))
        eliminate_constraints_from_increment!(Δu[Block(i)], p, solver_cache)
    end
end
eliminate_constraints_from_increment!(Δu, problem::NullProblem, solver_cache) = nothing

function residual_norm(solver_cache::NewtonRaphsonSolverCache, problem::AbstractCoupledProblem)
    val = 0.0
    for (i,p) ∈ enumerate((base_problems(problem)))
        val += residual_norm(solver_cache, p, Block(i))
    end
    return val
end

function eliminate_constraints_from_linearization!(solver_cache, problem::AbstractCoupledProblem)
    for (i,p) ∈ enumerate(base_problems(problem))
        eliminate_constraints_from_linearization_blocked!(solver_cache, problem, Block(i))
    end
end

function eliminate_constraints_from_linearization_blocked!(solver_cache, problem::AbstractCoupledProblem, i_::Block)
    @assert length(i_.n) == 1
    i = i_.n[1]
    hasproperty(base_problems(problem)[i], :ch) || return nothing
    ch = getch(base_problems(problem)[i])
    # TODO optimize this
    for j in 1:length(base_problems(problem))
        if i == j
            jacobian_block = getJ(solver_cache.op, Block((i,i)))
            # Eliminate diagonal entry only
            residual_block = @view solver_cache.residual[i_]
            apply_zero!(jacobian_block, residual_block, ch)
        else
            # Eliminate rows
            jacobian_block = getJ(solver_cache.op, Block((i,j)))
            jacobian_block[ch.prescribed_dofs, :] .= 0.0
            # Eliminate columns
            jacobian_block = getJ(solver_cache.op, Block((j,i)))
            jacobian_block[:, ch.prescribed_dofs] .= 0.0
        end
    end

    return nothing
end

function residual_norm(solver_cache::NewtonRaphsonSolverCache, problem::RSAFDQ2022TyingProblem, i::Block) 
    norm(solver_cache.residual[Block(2)])
end

eliminate_constraints_from_increment!(Δu, problem::RSAFDQ2022TyingProblem, solver_cache) = nothing
function eliminate_constraints_from_linearization!(solver_cache, problem::RSAFDQ20223DProblem)
    @unpack structural_problem = problem
    @unpack op = solver_cache
    ch = getch(structural_problem)
    # Eliminate residual
    residual_block = @view solver_cache.residual[Block(1)]
    # Elimiante diagonal
    # apply_zero!(getJ(op, Block(1,1)), residual_block, ch) # FIXME crashes
    apply!(getJ(op, Block(1,1)), ch)
    apply_zero!(residual_block, ch)
    # Eliminate rows
    getJ(op, Block((1,2)))[ch.prescribed_dofs, :] .= 0.0
    # Eliminate columns
    getJ(op, Block((2,1)))[:, ch.prescribed_dofs] .= 0.0
end


#######################################################################################
function nlsolve!(u::AbstractVector, problem::Union{AbstractSemidiscreteFunction,Thunderbolt.RSAFDQ20223DProblem}, solver_cache::NewtonRaphsonSolverCache, t)
    @unpack op, residual = solver_cache
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, u, residual, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(solver_cache, problem)
        # vtk_grid("newton-debug-$newton_itr", problem.structural_problem.dh) do vtk
        #     vtk_point_data(vtk, problem.structural_problem.dh, u[Block(1)])
        #     vtk_point_data(vtk, problem.structural_problem.dh, residual[Block(1)], :residual)
        # end
        residualnorm = residual_norm(solver_cache, problem)
        @info "Newton itr $newton_itr: ||r||=$residualnorm"
        if residualnorm < solver_cache.parameters.tol #|| (newton_itr > 0 && norm(Δu) < solver_cache.parameters.tol)
            break
        elseif newton_itr > solver_cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        elseif any(isnan.(residualnorm))
            @warn "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm"
            return false
        end

        @timeit_debug "solve" !solve_inner_linear_system!(Δu, solver_cache) && return false

        eliminate_constraints_from_increment!(Δu, problem, solver_cache)

        u .-= Δu # Current guess
    end
    return true
end

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

    # For debugging purposes
    # J11 = @view J[Block(1,1)]
    # r1 =  @view r[Block(1)]
    # J22 = @view J[Block(2,2)]
    # r2 =  @view r[Block(2)]
    # Δd = J11 \ r1
    # Δp = J22 \ r2
    # return [Δd; Δp]
end
inner_solve(J::BlockMatrix, r) = SparseMatrixCSC(J) \ r
inner_solve(J, r::BlockArray) = J \ Vector(r)

function solve_inner_linear_system!(Δu, solver_cache::NewtonRaphsonSolverCache)
    J = getJ(solver_cache.op)
    r = solver_cache.residual
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
