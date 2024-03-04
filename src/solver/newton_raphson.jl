"""
    update_linearization!(Jᵤ, residual, u)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` and its residual `F(u)` in 
preparation to solve for the increment `Δu` with the linear problem `J(u) Δu = F(u)`.
"""
update_linearization!(Jᵤ, residual, u) = error("Not overloaded")

"""
    update_linearization!(Jᵤ, u, problem)

Setup the linearized operator `Jᵤ(u)``.
"""
update_linearization!(Jᵤ, u) = error("Not overloaded")

"""
    update_residual!(residual, u, problem)

Evaluate the residual `F(u)` of the problem.
"""
update_residual!(residual, u) = error("Not overloaded")

"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T}
    # Convergence tolerance
    tol::T = 1e-8
    # Maximum number of iterations
    max_iter::Int = 100
end

mutable struct NewtonRaphsonSolverCache{OpType, ResidualType, T}
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonRaphsonSolver{T}
    #linear_solver_cache
end

function setup_operator(problem::QuasiStaticNonlinearProblem, solver::NewtonRaphsonSolver)
    @unpack dh, constitutive_model, face_models = problem
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."

    intorder = quadrature_order(problem, :displacement)
    qr = QuadratureRuleCollection(intorder)
    qr_face = FaceQuadratureRuleCollection(intorder)

    return AssembledNonlinearOperator(
        dh, :displacement, constitutive_model, qr, face_models, qr_face
    )
end

function setup_operator(problem::NullProblem, solver::NewtonRaphsonSolver{T}) where T
    # return NullOperator(
    #     solution_size(problem)
    # )
    return DiagonalOperator(ones(T, solution_size(problem)))
end

function setup_solver_caches(problem::QuasiStaticNonlinearProblem, solver::NewtonRaphsonSolver{T}) where {T}
    NewtonRaphsonSolverCache(setup_operator(problem, solver), Vector{T}(undef, solution_size(problem)), solver)
end

function setup_solver_caches(coupled_problem::CoupledProblem{<:Tuple{<:QuasiStaticNonlinearProblem,<:NullProblem}}, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack base_problems, couplers = coupled_problem
    op = BlockOperator((
        [i == j ? setup_operator(base_problems[i], solver) : NullOperator{T,solution_size(base_problems[j]),solution_size(base_problems[i])}()  for i in 1:length(base_problems) for j in 1:length(base_problems)]...,
    ))
    solution = mortar([
        Vector{T}(undef, solution_size(base_problems[i])) for i ∈ 1:length(base_problems)
    ])

    NewtonRaphsonSolverCache(op, solution, solver)
end

eliminate_constraints_from_linearization!(solver_cache, problem) = apply_zero!(solver_cache.op.J, solver_cache.residual, problem.ch)
eliminate_constraints_from_increment!(Δu, problem, solver_cache) = apply_zero!(Δu, problem.ch)
residual_norm(solver_cache::NewtonRaphsonSolverCache, problem) = norm(solver_cache.residual[Ferrite.free_dofs(problem.ch)])

function eliminate_constraints_from_increment!(Δu, problem::CoupledProblem, solver_cache)
    for (i,p) ∈ enumerate(problem.base_problems)
        eliminate_constraints_from_increment!(Δu[Block(i)], p, solver_cache)
    end
end
eliminate_constraints_from_increment!(Δu, problem::NullProblem, solver_cache) = nothing

function residual_norm(solver_cache::NewtonRaphsonSolverCache, problem::CoupledProblem)
    val = 0.0
    for (i,p) ∈ enumerate(problem.base_problems)
        val += residual_norm(solver_cache, p, Block(i))
    end
    return val
end

residual_norm(solver_cache::NewtonRaphsonSolverCache, problem, i::Block) = norm(solver_cache.residual[i][Ferrite.free_dofs(problem.ch)])
residual_norm(solver_cache::NewtonRaphsonSolverCache, problem::NullProblem, i::Block) = 0.0

function eliminate_constraints_from_linearization!(solver_cache, problem::CoupledProblem)
    for (i,p) ∈ enumerate(problem.base_problems)
        eliminate_constraints_from_linearization_blocked!(solver_cache, problem, Block(i))
    end
end

# TODO FIXME this only works if the problem to eliminate is in the first block
function eliminate_constraints_from_linearization_blocked!(solver_cache, problem::CoupledProblem, i::Block)
    if i.n[1] > 1
        if typeof(problem.base_problems[2]) != NullProblem
            @error "Block elimination not working for block $i"
        else
            return nothing 
        end
    end
    # TODO more performant block elimination
    apply_zero!(solver_cache.op.operators[1].J, solver_cache.residual[i], problem.base_problems[1].ch)
end

function solve!(u, problem, solver_cache::NewtonRaphsonSolverCache, t)
    @unpack op, residual = solver_cache
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(solver_cache.op, u, residual, t)

        @timeit_debug "elimination" eliminate_constraints_from_linearization!(solver_cache, problem)
        residualnorm = residual_norm(solver_cache, problem)
        @info newton_itr, residualnorm
        if residualnorm < solver_cache.parameters.tol
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
inner_solve(J::BlockMatrix, r::BlockArray) = SparseMatrixCSC(J) \ Vector(r)
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
