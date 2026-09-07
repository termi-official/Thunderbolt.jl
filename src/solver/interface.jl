abstract type AbstractSolver <: SciMLBase.AbstractDEAlgorithm end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

abstract type AbstractTimeSolverCache end

# TODO remove
getJ(op) = op.J

# Nonlinear
setup_stage_operator(f::NullFunction, solver::AbstractSolver, local_solver_cache, t₀) =
    NullOperator{Float64, solution_size(f), solution_size(f)}()

# Linear
# Unrolled to disambiguate
function setup_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::AssemblyStrategy{<:FullAssembly, <:ColoredScheduling, <:AbstractCPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractGPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::AssemblyStrategy{<:FullAssembly, <:ColoredScheduling, <:AbstractGPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end

function setup_operator(
    strategy,
    integrator::AbstractLinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    return setup_operator(strategy, integrator, dh)
end

# Bilinear
function setup_operator(
    strategy::Union{
        AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
        AssemblyStrategy{<:FullAssembly, <:ColoredScheduling, <:AbstractCPUDevice},
    },
    integrator::AbstractBilinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    setup_assembled_operator(strategy, integrator, solver.system_matrix_type, dh)
end

function setup_operator(
    strategy::AssemblyStrategy{<:FullAssembly, <:Any, <:AbstractGPUDevice},
    integrator::AbstractBilinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    setup_assembled_operator(strategy, integrator, solver.system_matrix_type, dh)
end

"""
    setup_assembled_operator(strategy, integrator, system_matrix_type, dh)

Materialize `integrator` against `dh` into the operator a solver requesting `system_matrix_type` can
step with.

The device that assembles and the format the solver wants its system matrix in are independent
choices -- the model carries `strategy`, the solver carries `system_matrix_type` -- and this is where
they meet. Where both live on the host they may still differ in *format*: the affine backward Euler
stage reads the mass and diffusion matrices only through `nonzeros`, and every host allocator emits
the same ordering for one dof handler, so the requested format is simply not needed here. Across a
device boundary that no longer holds, and the extension method returns a
[`MirroredBilinearOperator`](@ref) instead.

Where the strategy names a *device*, there is no mirror: the operator assembles straight into
`system_matrix_type`, so the format is threaded onto the operator specification and the two knobs
have to name the same one.
"""
function setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type,
    dh::AbstractDofHandler,
)
    setup_operator(strategy, integrator, dh)
end

@doc (@doc setup_assembled_operator)
function setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, <:Any, <:AbstractGPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type,
    dh::AbstractDofHandler,
)
    return setup_operator(_device_assembly_strategy(strategy, system_matrix_type), integrator, dh)
end

# The two knobs meet here. A device assembly writes the entries of `system_matrix_type` itself, so it
# has to be a format Ferrite ships a device assembler for -- CSC, not CSR -- and the operator
# specification is where `FerriteOperators` reads the type from.
function _device_assembly_strategy(
    strategy::AssemblyStrategy{<:FullAssembly, <:Any, <:AbstractGPUDevice},
    system_matrix_type::Type,
)
    spec = strategy.form.operator_specification
    # Anything but the standard specification is rejected for a device by FerriteOperators, with a
    # message naming the limitation. Hand it over untouched rather than rebuilding it into one.
    spec isa StandardOperatorSpecification || return strategy
    hasmethod(Ferrite.start_assemble, Tuple{system_matrix_type}) || error(
        "Cannot assemble on $(nameof(typeof(strategy.device))) into $system_matrix_type: Ferrite " *
        "has no device assembler for it. Device assembly needs a CSC device matrix -- a CSR one is " *
        "allocatable but not assemblable -- so pass `system_matrix_type = CuSparseMatrixCSC{Tv, Ti}` " *
        "to the solver. A CSR system matrix stays available with a host assembly strategy, which " *
        "assembles on the host and mirrors into the device matrix.",
    )
    declared = spec.matrix_type
    declared === nothing ||
        declared === system_matrix_type ||
        error(
            "The assembly strategy's operator specification names the matrix type $declared while " *
            "the solver asks for $system_matrix_type. The device assembles directly into the " *
            "solver's system matrix, so the two have to name the same type -- drop the one on the " *
            "specification.",
        )
    return AssemblyStrategy(
        FullAssembly(
            StandardOperatorSpecification(;
                algebraic_couplings = spec.algebraic_couplings,
                constraint_handler  = spec.constraint_handler,
                matrix_type         = system_matrix_type,
            ),
        ),
        strategy.scheduling,
        strategy.device,
    )
end

"""
    MirroredBilinearOperator(host_operator, A)

A bilinear operator assembled by `host_operator` on the host, whose matrix is mirrored into `A`.

`update_operator!` runs the host assembly and then refreshes `A`, so everything downstream --
`mul!`, and the `nonzeros` the affine backward Euler stage combines -- sees only `A`. That
correspondence is entrywise, so `A` has to be allocated by the same [`create_system_matrix`](@ref)
call as the stage matrix it is combined into; nothing here can check that beyond the entry count.
"""
struct MirroredBilinearOperator{OperatorType, MatrixType, BufferType} <: AbstractBilinearOperator
    host_operator::OperatorType
    A::MatrixType
    # Staging buffer in the mirror's value type. The host assembles in the strategy's value type,
    # which is not in general the solver's, and a cross-device `copyto!` needs the two to match.
    nzbuffer::BufferType
end

function MirroredBilinearOperator(host_operator, A)
    nnz_host   = length(nonzeros(host_operator.A))
    nnz_mirror = length(nonzeros(A))
    nnz_host == nnz_mirror || error(
        "Cannot mirror a $(nnz_host) entry matrix into a $(nnz_mirror) entry one. Both are " *
        "allocated from the same dof handler, so the two allocators disagree about the sparsity " *
        "pattern and the entrywise correspondence the mirror relies on does not exist.",
    )
    return MirroredBilinearOperator(
        host_operator,
        A,
        Vector{eltype(nonzeros(A))}(undef, nnz_mirror),
    )
end

function update_operator!(op::MirroredBilinearOperator, p, ctx = nothing)
    update_operator!(op.host_operator, p, ctx)
    op.nzbuffer .= nonzeros(op.host_operator.A)
    copyto!(nonzeros(op.A), op.nzbuffer)
    return nothing
end

mul!(out::AbstractVector, op::MirroredBilinearOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::MirroredBilinearOperator, in::AbstractVector, α, β) =
    mul!(out, op.A, in, α, β)

# Nonlinear
"""
    setup_stage_operator(f, solver, local_solver_cache, t₀)

The operator the stage of `solver` solves with, for the function `f`.

Dispatches on the **pair**, because what is assembled and which dof handler it is assembled against
are both properties of the combination: a continuation poses the internal forces alone, backward Euler
poses them with a condensed local problem underneath, and Newmark poses them together with an inertia
term. Each method therefore names its own handler; nothing infers one from `f`.

Distinct from the `setup_operator(strategy, integrator, dh)` family, which materializes one integrator
against one handler and knows nothing about who will solve with it.

`local_solver_cache` is the per-quadrature-point scratch from [`setup_local_solver_cache`](@ref), or
`nothing` where nothing is condensed. `t₀` is the time any operator that has to be assembled once at
setup is assembled at.
"""
function setup_stage_operator end

function update_constraints!(
    f::AbstractSemidiscreteFunction,
    solver_cache::AbstractTimeSolverCache,
    t,
)
    Ferrite.update!(getch(f), t)
    apply!(solver_cache.uₙ, getch(f))
end

update_constraints!(f, solver_cache::AbstractTimeSolverCache, t) = nothing

create_system_matrix(T::Type{<:AbstractMatrix}, f::AbstractSemidiscreteFunction) =
    create_system_matrix(T, f.dh)

# The CSC arrays are handed to the CSR type unchanged, so what this returns is the CSR of `Aᵀ`, and
# every operator assembled into it is assumed symmetric. Deliberate, not incidental: the entrywise
# `nonzeros` correspondence between this matrix and the CSC ones the operators own is what
# `_implicit_euler_heat_solver_update_system_matrix!` combines them through, and a faithful CSC→CSR
# conversion would reorder the entries and break it. The CUDA extension's
# `create_system_matrix(::Type{<:CuSparseMatrixCSR}, dh)` carries the same assumption for the same
# reason; the two have to be changed together.
function create_system_matrix(
    ::Type{<:ThreadedSparseMatrixCSR{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsct = transpose(convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh)))
    return ThreadedSparseMatrixCSR(Acsct)
end

function create_system_matrix(SpMatType::Type{<:SparseMatrixCSC}, dh::AbstractDofHandler)
    A = convert(SpMatType, allocate_matrix(dh))
    return A
end

function create_system_vector(::Type{<:Vector{T}}, f::AbstractSemidiscreteFunction) where {T}
    return zeros(T, solution_size(f))
end

function create_system_vector(::Type{<:Vector{T}}, dh::DofHandler) where {T}
    return zeros(T, ndofs(dh))
end
