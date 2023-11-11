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

function setup_solver_caches(problem::QuasiStaticNonlinearProblem, solver::NewtonRaphsonSolver{T}, t₀) where {T}
    @unpack dh, constitutive_model, face_models = problem
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the load stepper."

    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(dh.grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{ref_shape}(intorder)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    cv = CellValues(qr, ip, ip_geo)
    fv = FaceValues(qr_face, ip, ip_geo)

    # TODO abstraction layer around this! E.g. setup_element_cache(problem, solver)
    contraction_cache = setup_contraction_model_cache(cv, constitutive_model.contraction_model)
    element_cache = StructuralElementCache(
        constitutive_model,
        contraction_cache,
        cv
    )
    # TODO abstraction layer around this! E.g. setup_face_caches(problem, solver)
    face_caches = ntuple(i->setup_face_cache(face_models[i], fv, t₀), length(face_models))

    quasi_static_operator = AssembledNonlinearOperator(
        create_sparsity_pattern(dh),
        element_cache, 
        face_caches,
        dh,
    )

    NewtonRaphsonSolverCache(quasi_static_operator, Vector{Float64}(undef, solution_size(problem)), solver)
end

# TODO what is a cleaner solution for this?
function setup_solver_caches(coupled_problem::CoupledProblem{<:Tuple{<:QuasiStaticNonlinearProblem,<:NullProblem}}, solver::NewtonRaphsonSolver{T}, t₀) where {T}
    problem = coupled_problem.base_problems[1]
    @unpack dh, constitutive_model, face_models = problem
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the load stepper."

    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(dh.grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{ref_shape}(intorder)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    cv = CellValues(qr, ip, ip_geo)
    fv = FaceValues(qr_face, ip, ip_geo)

    # TODO abstraction layer around this! E.g. setup_element_cache(problem, solver)
    contraction_cache = setup_contraction_model_cache(cv, constitutive_model.contraction_model)
    element_cache = StructuralElementCache(
        constitutive_model,
        contraction_cache,
        cv
    )
    # TODO abstraction layer around this! E.g. setup_face_caches(problem, solver)
    face_caches = ntuple(i->setup_face_cache(face_models[i], fv, t₀), length(face_models))

    quasi_static_operator = AssembledNonlinearOperator(
        create_sparsity_pattern(dh),
        element_cache, 
        face_caches,
        dh,
    )
    error("Not implemented yet")
    # TODO introduce CouplingOperator
    # BlockOperator((
    #     quasi_static_operator, NullOperator{Float64,1,ndofs(dh)}(),
    #     NullOperator{Float64,ndofs(dh),1}(), NullOperator{Float64,1,1}()
    # ))
    op = BlockOperator((
        quasi_static_operator, NullOperator{Float64,1,ndofs(dh)}(),
        NullOperator{Float64,ndofs(dh),1}(), DiagonalOperator([1.0])
    ))
    solution = mortar([
        Vector{Float64}(undef, solution_size(coupled_problem.base_problems[1])),
        Vector{Float64}(undef, solution_size(coupled_problem.base_problems[2]))
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

function solve!(u, problem, solver_cache::NewtonRaphsonSolverCache{OpType, ResidualType, T}, t) where {OpType, ResidualType, T}
    @unpack op, residual = solver_cache
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        residual .= 0.0
        update_linearization!(solver_cache.op, u, residual, t)

        eliminate_constraints_from_linearization!(solver_cache, problem)
        residualnorm = residual_norm(solver_cache, problem)
        if residualnorm < solver_cache.parameters.tol
            break
        elseif newton_itr > solver_cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        end

        !solve_inner_linear_system!(Δu, solver_cache) && return false

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
