abstract type AbstractSolver end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

"""
    solve(problem, solver, Δt, time_span, initial_condition[, callback])

Main entry point for solvers in Thunderbolt.jl. The design is inspired by
DifferentialEquations.jl. We try to upstream as much content as possible to
make it available for packages.

TODO iterator syntax instead of callback
"""
function solve(problem::AbstractProblem, solver::AbstractSolver, Δt₀, (t₀, T), initial_condition, callback = (t,p,c) -> nothing)
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

function setup_operator(problem::QuasiStaticNonlinearProblem, solver::AbstractNonlinearSolver)
    @unpack dh, constitutive_model, face_models = problem
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    symbol_guess = dh.field_names[1]

    intorder = quadrature_order(problem, symbol_guess)
    qr = QuadratureRuleCollection(intorder)
    qr_face = FaceQuadratureRuleCollection(intorder)

    return AssembledNonlinearOperator(
        dh, symbol_guess, constitutive_model, qr, face_models, qr_face
    )
end

function setup_operator(problem::QuasiStaticNonlinearProblem, couplings, solver::AbstractNonlinearSolver)
    @unpack dh, constitutive_model, face_models = problem
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    symbol_guess = dh.field_names[1]

    intorder = quadrature_order(problem, symbol_guess)
    qr = QuadratureRuleCollection(intorder)
    qr_face = FaceQuadratureRuleCollection(intorder)

    # tying_models = []
    # for coupling in couplings
    #     if is_relevant_coupling(coupling)
    #         relevant_couplings(coupling)
    #     end
    # end

    # tying_models_ = length(tying_models) > 1 ? (tying_models...,) : tying_models[1]
    tying_models_ = []
    return AssembledNonlinearOperator(
        dh, symbol_guess, constitutive_model, qr, face_models, qr_face, tying_models_, qr_face,
    )
end

# TODO correct dispatches
function setup_coupling_operator(first_problem::AbstractProblem, second_problem::AbstractProblem, couplings, solver::AbstractNonlinearSolver)
    NullOperator{Float64,solution_size(second_problem),solution_size(first_problem)}()
end

# Block-Diagonal entry
setup_operator(coupled_problem::CoupledProblem, i::Int, solver) = setup_operator(coupled_problem.base_problems[i], coupled_problem.couplings, solver)
# Offdiagonal entry
setup_coupling_operator(coupled_problem::CoupledProblem, i::Int, j::Int, solver) = setup_coupling_operator(coupled_problem.base_problems[i], coupled_problem.base_problems[j], coupled_problem.couplings, solver)
