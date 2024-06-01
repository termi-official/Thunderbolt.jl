abstract type AbstractSolver end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

"""
    legacysolve(problem, solver, Δt, time_span, initial_condition[, callback])

Main entry point for solvers in Thunderbolt.jl. The design is inspired by
DifferentialEquations.jl. We try to upstream as much content as possible to
make it available for packages.
"""
function legacysolve(problem::DiffEqBase.AbstractDEProblem, solver::AbstractSolver, Δt₀, (t₀, T), initial_condition, callback = (t,p,c) -> nothing)
    solver_cache = setup_solver_cache(problem, solver, t₀)

    setup_initial_condition!(problem, solver_cache, initial_condition, t₀)

    Δt = Δt₀
    t = t₀
    while t < T
        @info t, Δt
        if !perform_step!(problem, solver_cache, t, Δt)
            @warn "Time step $t failed."
            return false
        end

        callback(t, problem, solver_cache)

        t, Δt = adapt_timestep(t, Δt, problem, solver_cache)
    end

    @info T
    if !perform_step!(problem, solver_cache, t, T-t)
        @warn "Time step $t failed."
        return false
    end

    callback(t, problem, solver_cache)

    return true
end

adapt_timestep(t, Δt, problem, solver_cache) = (t += Δt, Δt) # :)

"""
    setup_initial_condition!(problem, cache, initial_condition, time)

Main entry point to setup the initial condition of a problem for a given solver.
"""
function setup_initial_condition!(problem, cache, initial_condition, time)
    u₀ = initial_condition(problem, time)
    cache.uₙ .= u₀
    return nothing
end

function setup_operator(problem::NullProblem, solver)
    return DiagonalOperator([1.0])
    # return NullOperator{Float64,solution_size(problem),solution_size(problem)}()
end

function setup_operator(problem::NullProblem, couplings, solver)
    return DiagonalOperator([1.0])
    # return NullOperator{Float64,solution_size(problem),solution_size(problem)}()
end

function setup_operator(problem::QuasiStaticProblem, solver::AbstractNonlinearSolver)
    @unpack dh, constitutive_model, face_models = problem.f
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    displacement_symbol = first(dh.field_names)

    intorder = quadrature_order(problem.f, displacement_symbol)
    qr = QuadratureRuleCollection(intorder)
    qr_face = FaceQuadratureRuleCollection(intorder)

    return AssembledNonlinearOperator(
        dh, displacement_symbol, constitutive_model, qr, face_models, qr_face
    )
end

# function setup_operator(problem::QuasiStaticProblem, relevant_coupler, solver::AbstractNonlinearSolver)
#     @unpack dh, constitutive_model, face_models, displacement_symbol = problem
#     @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."
#     @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

#     intorder = quadrature_order(problem, displacement_symbol)
#     qr = QuadratureRuleCollection(intorder)
#     qr_face = FaceQuadratureRuleCollection(intorder)

#     return AssembledNonlinearOperator(
#         dh, displacement_symbol, constitutive_model, qr, face_models, qr_face, relevant_coupler, ???, <- depending on the coupler either face or element qr
#     )
# end

function setup_operator(problem::RSAFDQ20223DProblem, solver::AbstractNonlinearSolver)
    @unpack tying_problem, structural_problem = problem
    # @unpack dh, constitutive_model, face_models, displacement_symbol = structural_problem
    @unpack dh, constitutive_model, face_models = structural_problem.f
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    displacement_symbol = first(dh.field_names)

    intorder = quadrature_order(structural_problem, displacement_symbol)
    qr = QuadratureRuleCollection(intorder)
    qr_face = FaceQuadratureRuleCollection(intorder)

    return AssembledRSAFDQ2022Operator(
        dh, displacement_symbol, constitutive_model, qr, face_models, qr_face, tying_problem
    )
end

# TODO correct dispatches
function setup_coupling_operator(first_problem::DiffEqBase.AbstractDEProblem, second_problem::DiffEqBase.AbstractDEProblem, relevant_couplings, solver::AbstractNonlinearSolver)
    NullOperator{Float64,solution_size(second_problem),solution_size(first_problem)}()
end

# Block-Diagonal entry
setup_operator(coupled_problem::CoupledProblem, i::Int, solver) = setup_operator(coupled_problem.base_problems[i], coupled_problem.couplings, solver)
# Offdiagonal entry
setup_coupling_operator(coupled_problem::CoupledProblem, i::Int, j::Int, solver) = setup_coupling_operator(coupled_problem.base_problems[i], coupled_problem.base_problems[j], coupled_problem.couplings, solver)
