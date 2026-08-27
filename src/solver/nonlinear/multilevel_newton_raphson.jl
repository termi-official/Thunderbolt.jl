Base.@kwdef struct GenericLocalNonlinearSolver <: AbstractNonlinearSolver
    max_iters::Int = 10
    tol::Float64 = 1e-4
end

"""
    LocalSolveReport

Outcome of the local Newton at one quadrature point: its return code, its residual norm at exit and
the number of Newton passes it took. A point whose local problem is closed form records nothing and
keeps the default, `Default`/`0.0`/`0` -- not a failure, and no iterations to report.
"""
struct LocalSolveReport
    retcode::SciMLBase.ReturnCode.T
    residual::Float64
    iterations::Int
end
LocalSolveReport() = LocalSolveReport(SciMLBase.ReturnCode.Default, 0.0, 0)

_local_solve_failed(report::LocalSolveReport) =
    report.retcode ∉ (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)

"""
    setup_local_solve_reports(dh, lvh, ndofs_per_quadrature_point, cellset)

One report slot per quadrature point, in a single contiguous vector addressed per cell.

Each local solve writes exactly its own slot, so the store needs no counter and never grows -- which
is what makes it safe under threaded assembly. `duplicate_for_device` shares it rather than copying
it, so failures recorded by a worker remain visible to the global solver.

The flat layout is chosen with a device port in mind, but this is **not** GPU ready: nothing adapts
`DenseDataRange`, and a shared host vector is exactly what an `Adapt.adapt_structure` would have to
replace.

The per-cell quadrature point count is recovered from the [`InternalVariableHandler`](@ref), which
lays out `nqp * ndofs_per_quadrature_point` condensed unknowns per cell. `cellset` restricts the
store to the subdomain this solver serves; cells outside it get an empty range.

Dividing to recover `nqp` presumes one size for every quadrature point of the cell, the same assumption
the element layout and the local solver cache make; see [`internal_variable_size`](@ref) for the case that
breaks it.
"""
function setup_local_solve_reports(dh, lvh, ndofs_per_quadrature_point::Int, cellset)
    ndofs_per_quadrature_point == 0 && return nothing
    ncells     = getncells(get_grid(dh))
    offsets    = Vector{Int}(undef, ncells+1)
    offsets[1] = 1
    for cellid = 1:ncells
        ndofs_cell = length(FerriteOperators.internal_variable_range(lvh, cellid))
        nqp =
            (cellset === nothing || cellid ∈ cellset) ? ndofs_cell ÷ ndofs_per_quadrature_point : 0
        offsets[cellid+1] = offsets[cellid] + nqp
    end
    return DenseDataRange(fill(LocalSolveReport(), offsets[end]-1), offsets)
end

"""
Tag for the dual numbers of the local solve -- both the Newton's Jacobian and the corrector's
derivative. It exists so the configs carrying the dual work buffers can be built once at setup
instead of once per quadrature point solve, which is what `ForwardDiff` does when it is not handed
one. The two are never active at the same time: the corrector runs after the Newton has converged.

Naming the tag rather than deriving it from the residual closure is what makes that possible: the
closure is created per quadrature point, so a tag derived from it is not available at setup. It is a
custom tag in ForwardDiff's sense, so `checktag` passes it through.

The safety this gives up -- detection of perturbation confusion under nested differentiation -- the
local solve does not have to begin with: [`GenericLocalNonlinearSolverCache`](@ref) holds `Float64`
buffers, so a dual from an outer differentiation cannot reach this Newton in the first place.
"""
struct LocalSolveADTag end

"""
    GenericLocalNonlinearSolverCache

Immutable: everything that changes during a solve lives in the arrays it holds, the outer solver's
current tolerance in `outer_tol` and the per-quadrature-point outcomes in `reports`.

Immutability is a precondition for a device port, not a sufficient one -- the struct holds `Vector`s
and so is not `isbits`, and no `Adapt.adapt_structure` exists for it. What it buys today is that
`duplicate_for_device` can hand a worker a value rather than aliasing mutable solver state.
"""
Base.@kwdef struct GenericLocalNonlinearSolverCache{
    JacobianType,
    ResidualType,
    CorrectorRhsType,
    ReportsType,
    TolType,
    JacobianConfigType,
    DerivativeConfigType,
}
    params::GenericLocalNonlinearSolver
    J::JacobianType
    residual::ResidualType
    rhs_corrector::CorrectorRhsType
    reports::ReportsType = nothing
    outer_tol::TolType = [0.0]
    # Dual work buffers for the local Jacobian and for the corrector, see `LocalSolveADTag`.
    jacobian_config::JacobianConfigType = nothing
    derivative_config::DerivativeConfigType = nothing
end

"""
The `ForwardDiff.JacobianConfig` a local problem of `n` unknowns needs, tagged with
[`LocalSolveADTag`](@ref).
"""
setup_local_jacobian_config(residual::AbstractVector) = ForwardDiff.JacobianConfig(
    nothing,
    residual,
    residual,
    ForwardDiff.Chunk(residual),
    LocalSolveADTag(),
)

"""
The `ForwardDiff.DerivativeConfig` the corrector solves need -- they differentiate the same residual
with respect to a single scalar. Tagged with [`LocalSolveADTag`](@ref).
"""
setup_local_derivative_config(residual::AbstractVector{T}) where {T} =
    ForwardDiff.DerivativeConfig(nothing, residual, zero(T), LocalSolveADTag())

function duplicate_for_device(device, cache::GenericLocalNonlinearSolverCache)
    residual = duplicate_for_device(device, cache.residual)
    GenericLocalNonlinearSolverCache(;
        params        = cache.params,
        J             = duplicate_for_device(device, cache.J),
        residual      = residual,
        rhs_corrector = duplicate_for_device(device, cache.rhs_corrector),
        # Both are deliberately shared rather than copied: workers write disjoint report slots and a
        # failure must survive the worker, and the tolerance is written once per outer iteration and
        # has to reach every worker.
        reports   = cache.reports,
        outer_tol = cache.outer_tol,
        # Not shared: the configs *are* the scratch the derivatives are evaluated into, so two
        # workers sharing one would overwrite each other's duals.
        jacobian_config   = setup_local_jacobian_config(residual),
        derivative_config = setup_local_derivative_config(residual),
    )
end

"""
    record_local_solve!(local_solver_cache, cellid, qpi, retcode, residualnorm, iterations)

Record the outcome of one local solve in the shared per-quadrature-point store.
"""
@inline function record_local_solve!(
    local_solver_cache::GenericLocalNonlinearSolverCache,
    cellid,
    qpi,
    retcode,
    residualnorm,
    iterations,
)
    reports = local_solver_cache.reports
    reports === nothing && return nothing
    get_data_for_index(reports, cellid)[qpi] = LocalSolveReport(retcode, residualnorm, iterations)
    return nothing
end

"""
    local_solve_report(local_solver_cache, cellid, qpi)

The outcome recorded for one quadrature point of the current assembly pass.
"""
@inline function local_solve_report(
    local_solver_cache::GenericLocalNonlinearSolverCache,
    cellid,
    qpi,
)
    reports = local_solver_cache.reports
    reports === nothing && return LocalSolveReport()
    return get_data_for_index(reports, cellid)[qpi]
end

"""
    cell_condensation_report(local_solver_cache, cellid, nqp) -> CondensationReport

One cell's `nqp` local outcomes as the [`CondensationReport`](@ref) a condensation sweep folds:
`solves` counts the local problems the cell posed, `converged` is their conjunction, and the cell
names itself as `worst_cell` only where some point actually iterated -- `0` is the convention for
"no iterations anywhere", which is what a closed-form local solve reports.

`dt_factor` stays `1`: no stepper in this package is adaptive, so there is nobody to request a step
reduction from and a fabricated factor would be read as one.

Read after the sweep's own solves have written their slots, so what it folds is this pass.
"""
function cell_condensation_report(
    local_solver_cache::GenericLocalNonlinearSolverCache,
    cellid,
    nqp,
)
    converged        = true
    iterations       = 0
    worst_iterations = 0
    worst_qp         = 0
    worst_residual   = 0.0
    for qpi = 1:nqp
        report = local_solve_report(local_solver_cache, cellid, qpi)
        converged &= !_local_solve_failed(report)
        iterations += report.iterations
        worst_residual = max(worst_residual, report.residual)
        if report.iterations > worst_iterations
            worst_iterations = report.iterations
            worst_qp = qpi
        end
    end
    return CondensationReport{Float64}(
        converged,
        nqp,
        iterations,
        worst_iterations,
        worst_iterations > 0 ? cellid : 0,
        worst_qp,
        worst_residual,
        1.0,
    )
end

# A closed-form local problem is condensed without a local solver cache, so nothing recorded an
# outcome: `nqp` solves happened, none of them iterated.
cell_condensation_report(::Nothing, cellid, nqp) =
    CondensationReport{Float64}(true, nqp, 0, 0, 0, 0, 0.0, 1.0)

"""
    describe_local_solve_failures(local_solver_cache)

Every failing quadrature point of the last assembly pass, as `cell`/`qp` pairs with their local
residual norm and return code. Empty when nothing failed.

The detail behind [`CondensationReport`](@ref)'s summary: the report is a fold and carries one
argmax, while this walks the store the fold read and so can say how many points failed and where.
Only a solver owning that store — the multilevel Newton — can ask.
"""
function describe_local_solve_failures(local_solver_cache::GenericLocalNonlinearSolverCache)
    reports = local_solver_cache.reports
    reports === nothing && return ""
    io = IOBuffer()
    for cellid = 1:(length(reports.offsets)-1)
        for (qpi, report) in enumerate(get_data_for_index(reports, cellid))
            _local_solve_failed(report) || continue
            println(
                io,
                "  cell $cellid qp $qpi: $(report.retcode), ||r|| = $(report.residual) after $(report.iterations) iterations",
            )
        end
    end
    return String(take!(io))
end
function describe_local_solve_failures(local_solver_cache::Union{Tuple, AbstractVector})
    return join(describe_local_solve_failures.(local_solver_cache))
end

"""
    reset_local_solve_status!(local_solver_cache)

Clear the recorded outcomes before an assembly pass, so [`cell_condensation_report`](@ref) and
[`describe_local_solve_failures`](@ref) report on *that* pass alone.

Without this the failures latch, and the first one would poison every later assembly — including any
retry of the step, which is the only way a local failure can ever be recovered from.

Note that recovering additionally requires the time integrator to actually retry at a shorter `dt`.
`BackwardEulerSolver` does not: it is not adaptive, so a local failure currently ends the solve with
`ConvergenceFailure`.
"""
function reset_local_solve_status!(local_solver_cache::GenericLocalNonlinearSolverCache)
    reports = local_solver_cache.reports
    reports === nothing || fill!(reports.data, LocalSolveReport())
    return nothing
end
function reset_local_solve_status!(local_solver_cache::Tuple)
    foreach(reset_local_solve_status!, local_solver_cache)
end
function reset_local_solve_status!(local_solver_cache::AbstractVector)
    foreach(reset_local_solve_status!, local_solver_cache)
end

function set_local_solver_tol(local_solver_cache::GenericLocalNonlinearSolverCache, tol)
    local_solver_cache.outer_tol[1] = tol
end
function set_local_solver_tol(local_solver_cache::Tuple, tol)
    set_local_solver_tol.(local_solver_cache, tol)
end
function set_local_solver_tol(local_solver_cache::AbstractVector, tol)
    set_local_solver_tol.(local_solver_cache, tol)
end

"""
    MultilevelNewtonRaphsonSolver{T}

Multilevel Newton-Raphson solver [RabSanHsu:1979:mna](@ref) for nonlinear problems of the form `F(u,v) = 0; G(u,v) = 0`.

Like [`NewtonRaphsonSolver`](@ref) it solves an [`AbstractStageFunction`](@ref), so what it needs
from a problem is that stage's [`update_stage_linearization!`](@ref) and — when the global Newton
runs with `simplified_newton = true` — [`evaluate_stage_residual!`](@ref).

The global Newton's `simplified_newton` and `forcing` settings apply here as they do to the plain
[`NewtonRaphsonSolver`](@ref). Note what a simplified step does *not* skip: the condensation phase,
because the residual is a function of the condensed state. What it reuses is the global Jacobian.
"""
Base.@kwdef struct MultiLevelNewtonRaphsonSolver{gSolverType <: NewtonRaphsonSolver, lSolverType} <:
                   AbstractNonlinearSolver
    newton::gSolverType = NewtonRaphsonSolver()
    local_solver::lSolverType = GenericLocalNonlinearSolver()
end

struct MultiLevelNewtonRaphsonSolverCache{gCacheType, lCacheType} <: AbstractNonlinearSolverCache
    global_solver_cache::gCacheType
    local_solver_cache::lCacheType
end

# `Θks` and the Newton parameters live on the global cache this one wraps.
global_newton_cache(cache::MultiLevelNewtonRaphsonSolverCache) = cache.global_solver_cache

function Base.show(io::IO, cache::MultiLevelNewtonRaphsonSolverCache)
    println(io, "MultiLevelNewtonRaphsonSolverCache:")
    Base.show(io, cache.global_solver_cache)
    if cache.local_solver_cache isa Tuple
        for local_solver_cache in cache.local_solver_cache
            Base.show(io, local_solver_cache)
        end
    else
        Base.show(io, cache.local_solver_cache)
    end
end

function nlsolve!(
    u::AbstractVector,
    sf::AbstractStageFunction,
    mlcache::MultiLevelNewtonRaphsonSolverCache,
    t,
)
    cache = mlcache.global_solver_cache
    f = getfunction(sf)
    op = getoperator(sf)

    @unpack residual, linear_solver_cache, Θks = cache
    monitor = cache.parameters.monitor
    simplified = cache.parameters.simplified_newton
    cache.iter = -1
    Δu = linear_solver_cache.u
    residualnormprev = 0.0
    Θ1prev = length(Θks) > 0 ? first(Θks) : 0.0
    resize!(Θks, 0)
    set_local_solver_tol(mlcache.local_solver_cache, 0.0)
    while true
        cache.iter += 1
        residual .= 0.0
        reset_local_solve_status!(mlcache.local_solver_cache)
        stage_ok = if simplified && cache.iter > 0
            # Simplified Newton: reuse the Jacobian and preconditioner from iteration 0. The local
            # problems are still solved -- the condensed state is what the residual is a function of
            # -- only their sensitivities are not, since no tangent is requested.
            @timeit_debug "update residual" evaluate_stage_residual!(sf, residual, u)
        else
            @timeit_debug "update operator" update_stage_linearization!(sf, residual, u)
        end
        if !stage_ok
            # `condense_stage!` has already named the worst offender from the folded report. This
            # solver additionally holds the per-quadrature-point store the fold summarised, so it can
            # list every failing point -- how many and where, which one argmax cannot say.
            @debug "Local solve failures of this pass:\n$(describe_local_solve_failures(mlcache.local_solver_cache))" _group =
                :nlsolve
            return false
        end
        if simplified && cache.iter > 0
            @timeit_debug "elimination" eliminate_constraints_from_residual!(cache, sf)
            # Leave isfresh / precsisfresh false → reuse the existing factorization.
        else
            @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, sf)
            # Both flags: the matrix changed, and so must anything built from it. Setting only
            # `isfresh` left a `precs` preconditioner built once from the numerically empty Jacobian
            # and never rebuilt, because Thunderbolt mutates `op.J` in place and LinearSolve raises
            # `precsisfresh` only on a `setproperty!` of `A`.
            linear_solver_cache.isfresh = true
            linear_solver_cache.precsisfresh = true
        end

        residualnorm = residual_norm(cache, sf)
        set_local_solver_tol(mlcache.local_solver_cache, residualnorm^2)
        if residualnorm < cache.parameters.tol && cache.iter > 1 # Do at least two iterations to get a sane convergence estimate
            break
        elseif cache.iter > cache.parameters.max_iter
            @debug "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        elseif any(isnan.(residualnorm))
            @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        end

        _ew_prestep!(cache.forcing_cache, linear_solver_cache, residualnorm, cache.iter)
        # See the note in the plain Newton: the Eisenstat-Walker criterion is relative to ‖r₀‖, so a
        # warm-started increment would satisfy it trivially.
        cache.forcing_cache !== nothing && fill!(Δu, zero(eltype(Δu)))
        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        nonlinear_step_monitor(cache, t, f, u, cache.parameters.monitor)
        solve_succeeded =
            LinearSolve.SciMLBase.successful_retcode(sol) ||
            sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, sf, cache)

        # Only the entries the linear system solves for; the condensed tail is written by the
        # assembly, not by the increment.
        @inbounds @views u[uncondensed_range(sf)] .-= Δu

        if cache.iter > 0
            # In this case we might be unablet to estimate the convergence rate, because we are too close to the solution
            if residualnormprev < cache.parameters.tol && residualnorm < cache.parameters.tol
                push!(Θks, Θ1prev^2)
                break
            end
            Θk = residualnorm/residualnormprev
            push!(Θks, isnan(Θk) ? 0.0 : Θk)
            # A simplified Newton converges linearly, so a rate close to one is expected rather than
            # a symptom -- hence the same opt-out the plain Newton has.
            if cache.parameters.enforce_monotonic_convergence && Θk ≥ 1.0
                @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
                return false
            end

            # Late out on second iteration
            if residualnorm < cache.parameters.tol
                break
            end
        end

        residualnormprev = residualnorm
    end
    nonlinear_finalize_monitor(cache, t, f, monitor)
    return true
end
