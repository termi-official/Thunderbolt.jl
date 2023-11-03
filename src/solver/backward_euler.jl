#########################################################
########################## TIME #########################
#########################################################
struct BackwardEulerSolver
end

# TODO decouple from heat problem.
mutable struct BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, SourceTermType, LinSolverType, RHSType}
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    J::DiffusionMatrixType
    # Buffer for (M - Δt J)
    A::SystemMatrixType
    # Helper for possible source terms
    source_term::SourceTermType
    # Linear solver for (M - Δtₙ₋₁ J) uₙ = M uₙ₋₁
    linsolver::LinSolverType
    # Buffer for right hand side
    b::RHSType
    # Last time step length as a check if we have to update J
    Δt_last::Float64
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    cache.A = SystemMatrixType(cache.M.A - Δt*cache.J.A) # TODO FIXME make me generic
    cache.Δt_last = Δt
end

# Optimized version for CSR matrices
#TODO where is AbstractSparseMatrixCSR ?
# function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache{SolutionType, SMT1, SMT2, SMT2, LinSolverType, RHSType}, Δt) where {SolutionType, SMT1, SMT2, LinSolverType, RHSType}
#     cache.A = SparseMatrixCSR(transpose(cache.M.M - Δt*cache.J.K)) # TODO FIXME make me generic
# end

function implicit_euler_heat_update_source_term!(cache::BackwardEulerSolverCache, t)
    update_operator!(cache.source_term, t)
end

# Performs a backward Euler step
# TODO check if operator is time dependent and update
function perform_step!(problem::TransientHeatProblem, cache::BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, t, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    @unpack Δt_last, b, M, A, uₙ, uₙ₋₁, linsolver = cache
    # Remember last solution
    @inbounds uₙ₋₁ .= uₙ
    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(cache, Δt)
    # Prepare right hand side b = M uₙ₋₁
    mul!(b, M, uₙ₋₁)
    # TODO How to remove these two lines here?
    # Update source term
    implicit_euler_heat_update_source_term!(cache, t)
    add!(b, cache.source_term)
    # Solve linear problem
    # TODO abstraction layer and way to pass the solver/preconditioner pair (LinearSolve.jl?)
    Krylov.cg!(linsolver, A, b, uₙ₋₁)
    @inbounds uₙ .= linsolver.x
    @info linsolver.stats
    return true
end

function setup_solver_caches(problem::TransientHeatProblem, solver::BackwardEulerSolver, t₀)
    @unpack dh = problem
    @assert length(dh.field_names) == 1 # TODO relax this assumption, maybe.
    ip = dh.subdofhandlers[1].field_interpolations[1]
    order = Ferrite.getorder(ip)
    refshape = Ferrite.getrefshape(ip)
    sdim = Ferrite.getdim(dh.grid)
    qr = QuadratureRule{refshape}(2*order) # TODO how to pass this one down here?
    cv = CellValues(qr, ip)

    # TODO abstraction layer around AssembledBilinearOperator
    mass_operator = AssembledBilinearOperator(
        create_sparsity_pattern(dh),
        BilinearMassElementCache(
            BilinearMassIntegrator(
                ConstantCoefficient(1.0)
            ),
            zeros(getnquadpoints(qr)),
            cv
        ),
        dh,
    )

    # TODO abstraction layer around AssembledBilinearOperator
    diffusion_operator = AssembledBilinearOperator(
        create_sparsity_pattern(dh),
        BilinearDiffusionElementCache(
            BilinearDiffusionIntegrator(
                problem.diffusion_tensor_field,
            ),
            cv,
        ),
        dh,
    )

    cache = BackwardEulerSolverCache(
        zeros(ndofs(dh)),
        zeros(ndofs(dh)),
        # TODO How to choose the exact operator types here?
        #      Maybe via some parameter in BackwardEulerSolver?
        mass_operator,
        diffusion_operator,
        create_sparsity_pattern(dh),
        create_linear_operator(dh, problem.source_term),
        # TODO this via LinearSolvers.jl?
        CgSolver(
            ndofs(dh),
            ndofs(dh),
            Vector{Float64}
        ),
        zeros(ndofs(dh)),
        0.0
    )

    update_operator!(cache.M, t₀)
    update_operator!(cache.J, t₀)
    
    return cache
end

struct ForwardEulerSolver <: AbstractPointwiseSolver
end

mutable struct ForwardEulerSolverCache{VT,F} <: AbstractPointwiseSolverCache
    du::VT
    uₜ::VT
    rhs!::F
end

function perform_step!(cell_model::ION, t::Float64, Δt::Float64, solver_cache::ForwardEulerSolverCache)
    @unpack du, uₜ, rhs! = solver_cache
    @inbounds rhs!(du, uₜ, t)
    @inbounds uₜ[i] = φₘ_cell + Δt*du[1]

    return true
end

function setup_solver_caches(problem, solver::ForwardEulerSolver, t₀)
    return ForwardEulerSolverCache(
        zeros(num_states(problem.ode)),
        zeros(num_states(problem.ode))
    )
end
