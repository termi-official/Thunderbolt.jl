#########################################################
########################## TIME #########################
#########################################################
struct BackwardEulerSolver
end

# TODO decouple from heat problem via special ODEFunction (AffineODEFunction)
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
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache{<:Any, <:Any, <:Any, SystemMatrixType}, Δt) where {SystemMatrixType}
    cache.A = SystemMatrixType(cache.M.A - Δt*cache.J.A) # TODO FIXME make me generic
    cache.Δt_last = Δt
end
# Optimized version for CSR matrices - Lucky gamble dispatch
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache{<:Any, <:Any, <:Any, SystemMatrixType}, Δt) where {SystemMatrixType <: ThreadedSparseMatrixCSR}
    # We live in a symmeric utopia :)
    @timeit_debug "implicit_euler_heat_solver_update_system_matrix!" cache.A = SystemMatrixType(transpose(cache.M.A - Δt*cache.J.A)) # TODO FIXME make me generic
    cache.Δt_last = Δt
end

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
    @timeit_debug "b = M uₙ₋₁" mul!(b, M, uₙ₋₁)
    # TODO How to remove these two lines here?
    # Update source term
    @timeit_debug "update source term" begin
        implicit_euler_heat_update_source_term!(cache, t)
        add!(b, cache.source_term)
    end
    # Solve linear problem
    # TODO abstraction layer and way to pass the solver/preconditioner pair (LinearSolve.jl?)
    @timeit_debug "inner solve" Krylov.cg!(linsolver, A, b, uₙ₋₁)
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
        zeros(solution_size(problem)),
        zeros(solution_size(problem)),
        # TODO How to choose the exact operator types here?
        #      Maybe via some parameter in BackwardEulerSolver?
        mass_operator,
        diffusion_operator,
        ThreadedSparseMatrixCSR(transpose(create_sparsity_pattern(dh))), # TODO this should be decided via some interface
        create_linear_operator(dh, problem.source_term),
        # TODO this via LinearSolvers.jl?
        CgSolver(
            solution_size(problem),
            solution_size(problem),
            Vector{Float64}
        ),
        zeros(solution_size(problem)),
        0.0
    )

    @timeit_debug "initial assembly" begin
        update_operator!(cache.M, t₀)
        update_operator!(cache.J, t₀)
    end
    
    return cache
end

struct ForwardEulerSolver
end

mutable struct ForwardEulerSolverCache{VT,F}
    du::VT
    uₙ::VT
    rhs!::F
end

function perform_step!(problem, solver_cache::ForwardEulerSolverCache, t::Float64, Δt::Float64)
    @unpack du, uₙ, rhs! = solver_cache
    @inbounds rhs!(du, uₙ, t, problem.p)
    @inbounds uₙ .= uₙ .+ Δt .* du

    return true
end

function setup_solver_caches(problem::ODEProblem, solver::ForwardEulerSolver, t₀)
    return ForwardEulerSolverCache(
        zeros(num_states(problem.ode)),
        zeros(num_states(problem.ode)),
        problem.f
    )
end

