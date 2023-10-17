#########################################################
########################## TIME #########################
#########################################################
struct BackwardEulerSolver
end

mutable struct BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
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

# Performs a backward Euler step
# TODO check if operator is time dependent and update
function perform_step!(problem, cache::BackwardEulerSolverCache{SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}, t, Δt) where {SolutionType, MassMatrixType, DiffusionMatrixType, SystemMatrixType, LinSolverType, RHSType}
    @unpack Δt_last, b, M, A, uₙ, uₙ₋₁, linsolver = cache
    # Remember last solution
    @inbounds uₙ₋₁ .= uₙ
    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(cache, Δt)
    # Prepare right hand side b = M uₙ₋₁
    mul!(b, M, uₙ₋₁)
    # Solve linear problem
    # TODO abstraction layer and way to pass the solver/preconditioner pair (LinearSolve.jl?)
    Krylov.cg!(linsolver, A, b, uₙ₋₁)
    @inbounds uₙ .= linsolver.x

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

    diffusion_operator = AssembledBilinearOperator(
        create_sparsity_pattern(dh),
        BilinearDiffusionElementCache(
            BilinearDiffusionIntegrator(
                problem.diffusion_tensor_field
            ),
            zeros(Tensor{2,sdim}, getnquadpoints(qr)),
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
