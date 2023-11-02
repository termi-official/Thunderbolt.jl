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
    contraction_cache = Thunderbolt.setup_contraction_model_cache(cv, constitutive_model.contraction_model)
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

    NewtonRaphsonSolverCache(quasi_static_operator, Vector{Float64}(undef, ndofs(dh)), solver)
end

eliminate_constraints_from_linearization!(solver_cache, problem) = apply_zero!(solver_cache.op.J, solver_cache.residual, problem.ch)
eliminate_constraints_from_increment!(Δu, problem, solver_cache) = apply_zero!(Δu, problem.ch)
residual_norm(solver_cache::NewtonRaphsonSolverCache, problem) = norm(solver_cache.residual[Ferrite.free_dofs(problem.ch)])

function solve!(u, problem, solver_cache::NewtonRaphsonSolverCache{OpType, ResidualType, T}, t) where {OpType, ResidualType, T}
    @unpack op, residual = solver_cache
    newton_itr = -1
    Δu = zero(u)
    while true
        newton_itr += 1

        update_linearization!(solver_cache.op, u, residual, t)

        eliminate_constraints_from_linearization!(solver_cache, problem)
        residualnorm = residual_norm(solver_cache, problem)
        if residualnorm < solver_cache.parameters.tol
            break
        elseif newton_itr > solver_cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        end

        try
            solve_inner_linear_system!(Δu, solver_cache)
        catch err
            @warn "Linear solver failed: " , err
            return false
        end

        eliminate_constraints_from_increment!(Δu, problem, solver_cache)

        u .-= Δu # Current guess
    end
    return true
end

function solve_inner_linear_system!(Δu, solver_cache::NewtonRaphsonSolverCache{JacType, ResidualType, T}) where {JacType, ResidualType, T}
    Δu .= solver_cache.op.J \ solver_cache.residual
end
