"""
    Plonsey1964ECGIntegrator(diffusion::BilinearDiffusionIntegrator, κ∇φₘ)

The element description behind [`Plonsey1964ECGGaussCache`](@ref). It carries the diffusion
element's values and conductivity configuration — the ECG element is built from it, so both share
one `CellValues` and one coefficient cache — together with the per-quadrature-point flux buffer
`κ∇φₘ` that the cache's two sweeps write and read.

Evaluation only: it belongs to no integrator family, and its operator
(`setup_evaluation_operator`) holds the engine and no payload.
"""
struct Plonsey1964ECGIntegrator{
    DiffusionIntegratorType <: BilinearDiffusionIntegrator,
    QVectorType,
}
    diffusion::DiffusionIntegratorType
    κ∇φₘ::QVectorType
end

"""
The cache associated with [`Plonsey1964ECGIntegrator`](@ref). Serves the per-quadrature-point flux
sweep and the electrode integral ([`ElectrodePotentialFunctional`](@ref)) off the diffusion element
cache it composes; `κ∇φₘ` is the operator-wide flux buffer, shared unduplicated across workers
because the two sweeps touch disjoint cell slices of it.
"""
struct Plonsey1964ECGElementCache{DiffusionCacheType, QVectorType} <: AbstractVolumetricElementCache
    diffusion::DiffusionCacheType
    κ∇φₘ::QVectorType
end

setup_element_cache(integrator::Plonsey1964ECGIntegrator, sdh::SubDofHandler) =
    Plonsey1964ECGElementCache(
        setup_element_cache(integrator.diffusion, sdh),
        integrator.κ∇φₘ,
    )

duplicate_for_device(device, cache::Plonsey1964ECGElementCache) =
    Plonsey1964ECGElementCache(duplicate_for_device(device, cache.diffusion), cache.κ∇φₘ)

Ferrite.getnquadpoints(cache::Plonsey1964ECGElementCache) = getnquadpoints(cache.diffusion)
FerriteOperators.reinit_values!(cache::Plonsey1964ECGElementCache, cell) =
    reinit_values!(cache.diffusion, cell)

# Mandatory on every element cache, evaluation operator or not: the residual kernel is what setup
# validates for. The ECG element adds evaluation hooks to the diffusion element and changes none of
# its forms, so the residual is the composed element's — and no sweep of this operator issues it.
FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    cache::Plonsey1964ECGElementCache,
    args::CellArgs,
) = assemble_cell!(req, cache.diffusion, args)

# κ(x̃, t)∇φₘ(x̃) at one quadrature point -- the integrand of the Plonsey electrode integral below.
# A sweep carrying no context evaluates the conductivity at t = 0, which is what the Plonsey
# simplifications assume; a transient conductivity is reached by handing `update_ecg!` the context.
function _plonsey_quadrature_flux(φₘₑ, qp::Int, cell, cache::Plonsey1964ECGElementCache, pₑ, ctx)
    (; Dcache, cellvalues) = cache.diffusion
    κ = evaluate_coefficient(
        Dcache,
        cell,
        QuadraturePoint(qp, Ferrite.getpoints(cellvalues.qr)[qp]),
        _plonsey_time(ctx),
    )
    return κ ⋅ function_gradient(cellvalues, qp, φₘₑ)
end

_plonsey_time(::Nothing) = 0.0
_plonsey_time(ctx) = evaluation_time(ctx)

@doc raw"""
    ElectrodePotentialFunctional(x::Vec)

The reduction

```math
\int_\Omega \frac{\kappa \nabla \varphi_\mathrm{m}(\tilde{x}) \cdot (\tilde{x}-x)}{||\tilde{x}-x||^3} \mathrm{d}\tilde{x}
```

over the cells of a [`Plonsey1964ECGIntegrator`](@ref) operator, for the electrode at `x`. One
sweep evaluates one electrode. The integral is volumetric, so the kind is declared over the cell
family alone — an operator without the ECG element's cells fails the reduction's precondition
instead of answering a silent zero.

Reads the flux the last [`update_ecg!`](@ref) stored; evaluate through [`evaluate_ecg`](@ref),
which applies the ``-1/(4\pi\kappa_\mathrm{t})`` prefactor.
"""
struct ElectrodePotentialFunctional{dim, T}
    x::Vec{dim, T}
end

FerriteOperators.reduction_families(::Type{<:ElectrodePotentialFunctional}) = (:cells,)
FerriteOperators.functional_value_type(::ElectrodePotentialFunctional{dim, T}) where {dim, T} = T

function FerriteOperators.evaluate_cell_functional(
    kind::ElectrodePotentialFunctional{dim, T},
    cache::Plonsey1964ECGElementCache,
    args::CellArgs,
) where {dim, T}
    cv     = cache.diffusion.cellvalues
    coords = getcoordinates(args.cell)
    κ∇φₘₑ  = get_range_for_cell(cache.κ∇φₘ, cellid(args.cell))
    φₑ     = zero(T)
    @inbounds for qp = 1:getnquadpoints(cv)
        r = spatial_coordinate(cv, qp, coords) - kind.x
        φₑ += κ∇φₘₑ[qp] ⋅ r/norm(r)^3 * getdetJdV(cv, qp)
    end
    return φₑ
end

"""
    Plonsey1964ECGGaussCache(op::AbstractBilinearOperator, φₘ::AbstractVector)

Here φₘ is the solution vector containing the transmembranepotential, op is the associated diffusion opeartor and
κₜ is the torso's conductivity.

Returns a cache to compute the lead field with the form proposed in [Plo:1964:vcf](@cite)
with the Gauss theorem applied to it, as for example described in [OgiBalPer:2021:ema](@cite).
Calling [`evaluate_ecg`](@ref) with this method simply evaluates the following integral efficiently:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_t} \\int_\\Omega \\frac{ \\kappa_ ∇φₘ \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

The important simplifications taken are:
   1. Surrounding volume is an infinite, homogeneous sphere with isotropic conductivity
   2. The extracellular space and surrounding volume share the same isotropic, homogeneous conductivity tensor

The cache owns a payload-free [`Plonsey1964ECGIntegrator`](@ref) operator over `op`'s dof handler,
configured from `op`'s diffusion element. Both the flux sweep ([`update_ecg!`](@ref)) and the
electrode integral ([`evaluate_ecg`](@ref)) run through it; nothing is assembled.
"""
struct Plonsey1964ECGGaussCache{BufferType, OperatorType}
    # Buffer for storing "κ(x) ∇φₘ(x,t)" at the quadrature points
    κ∇φₘ::BufferType
    op::OperatorType
end

function Plonsey1964ECGGaussCache(op::BilinearFerriteOperator, φₘ::AbstractVector{T}) where {T}
    dh = op.engine.dh
    @assert length(dh.field_names) == 1 "Multiple fields detected. Problem setup might be broken..."
    sdim  = Ferrite.getspatialdim(get_grid(dh))
    κ∇φₘ  = setup_qvector(Vec{sdim, T}, dh, op.integrator.qrc)
    cache = Plonsey1964ECGGaussCache(
        κ∇φₘ,
        setup_evaluation_operator(
            op.engine.strategy,
            Plonsey1964ECGIntegrator(op.integrator, κ∇φₘ),
            dh,
        ),
    )
    update_ecg!(cache, φₘ)
    return cache
end

"""
    evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)

Compute the pseudo ECG at a given point x by evaluating:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_t} \\int_\\Omega \\frac{ \\kappa_ ∇φₘ \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

For more information please read the docstring for [`Plonsey1964ECGGaussCache`](@ref)
"""
function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)
    φₑ = FerriteOperators.evaluate_functional(
        method.op,
        ElectrodePotentialFunctional(x),
        (;),
        nothing,
    )
    return -φₑ / (4π*κₜ)
end

function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::AbstractVector{<:Vec}, κₜ::Real)
    φₑ = zeros(length(x))
    for i in eachindex(x)
        φₑ[i] = evaluate_ecg(method, x[i], κₜ)
    end
    return φₑ
end

"""
    update_ecg!(cache::Plonsey1964ECGGaussCache, φₘ::AbstractVector, ctx = nothing)

Refill the per-quadrature-point flux buffer `κ∇φₘ` from the transmembrane potential `φₘ`, in one
sweep of the cache's operator. `ctx` is what the conductivity is evaluated at; without one it is
the stationary `t = 0`.
"""
function update_ecg!(cache::Plonsey1964ECGGaussCache, φₘ::AbstractVector, ctx = nothing)
    evaluate_quadrature!(cache.κ∇φₘ, cache.op, φₘ, nothing, _plonsey_quadrature_flux; ctx)
    return nothing
end

"""
    PoissonECGReconstructionCache(fₑₚ::GenericSplitFunction, Ωₜ::AbstractMesh, κᵢ, κ, electrodes::AbstractVector{<:Vec}; ground, linear_solver, solution_vector_type, system_matrix_type)

Sets up a cache for calculating ``\\varphi_\\mathrm{e}`` by solving the Poisson problem
```math
\\nabla \\cdot (\\boldsymbol{\\kappa}_{\\mathrm{i}} + \\boldsymbol{\\kappa}_{\\mathrm{e}}) \\nabla \\varphi_{\\mathrm{e}}=-\\nabla \\cdot\\left(\\boldsymbol{\\kappa}_{\\mathrm{i}} \\nabla \\varphi_\\mathrm{m}\\right)
```
as for example proposed in [PotDubRicVinGul:2006:cmb](@cite) and investigated in [OgiBalPer:2021:ema](@cite) (as well as other studies). Here κₑ is the extracellular conductivity tensor and κᵢ is the intracellular conductivity tensor. The cache includes the assembled
stiffness matrix with applied homogeneous Dirichlet boundary condition at the first vertex of the mesh. As the problem is solved for each timestep with only the right hand side changing.

## Keyword Arguments
* `ground               = Set([VertexIndex(1, 1)])`
* `linear_solver        = LinearSolve.KrylovJL_CG()`
* `solution_vector_type = Vector{Float64}`
* `system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64}`

"""
struct PoissonECGReconstructionCache{
    DiffusionOperatorType1,
    DiffusionOperatorType2,
    TransferOperatorType,
    SolutionVectorType,
    SolverCacheType,
    PHType,
    CHType,
}
    torso_op::DiffusionOperatorType1  # Operator on the torso mesh for ∇κ∇
    source_op::DiffusionOperatorType2  # Operator on the heart mesh for ∇κᵢ∇
    transfer_op::TransferOperatorType # Transfer from heart to torso mesh
    ϕₑ::SolutionVectorType            # Solution vector buffer
    φₘ_t::SolutionVectorType          # Solution vector buffer on torso
    κ∇φₘ_t::SolutionVectorType        # Source term buffer on torso
    inner_solver::SolverCacheType     # Linear solver
    ph::PHType                        # PointEvalHandler on the torso
    ch::CHType                        # ConstraintHandler on the torso
end

# Convenience ctor to unpack default setup
function PoissonECGReconstructionCache(
    epfun::GenericSplitFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    torso_diffusion_tensor_field, # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vec};
    ground               = Set([VertexIndex(1, 1)]),
    torso_heart_domain   = nothing,
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64, Int64},
)
    PoissonECGReconstructionCache(
        epfun.functions[1],
        torso_grid,
        heart_diffusion_tensor_field,
        torso_diffusion_tensor_field,
        electrode_positions;
        ground,
        torso_heart_domain,
        linear_solver,
        solution_vector_type,
        system_matrix_type,
    )
end

function PoissonECGReconstructionCache(
    heart_fun::AffineODEFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    torso_diffusion_tensor_field, # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vec};
    ipc = LagrangeCollection{1}(),
    qrc = QuadratureRuleCollection(2),
    ground = OrderedSet([VertexIndex(1, 1)]),
    linear_solver = LinearSolve.KrylovJL_CG(),
    torso_heart_domain = nothing,
    solution_vector_type = Vector{Float64},
    system_matrix_type = ThreadedSparseMatrixCSR{Float64, Int64},
    extracellular_potential_symbol = :φₑ,
    strategy = SequentialAssemblyStrategy(PolyesterDevice()),
)
    heart_dh = heart_fun.dh
    heart_grid = get_grid(heart_dh)
    length(heart_dh.field_names) == 1 || @warn "Multiple fields detected. Setup might be broken..."

    torso_model = SteadyDiffusionModel(
        torso_diffusion_tensor_field,
        NoStimulationProtocol(), #ConstantCoefficient(NaN), # FIXME Poisoning to detecte if we accidentally touch these
        extracellular_potential_symbol,
    )

    torso_fun = semidiscretize(
        Dict(name => torso_model for name in subdomain_names(torso_grid)),
        FiniteElementDiscretization(
            Dict(extracellular_potential_symbol => ipc);
            dbcs = [Dirichlet(extracellular_potential_symbol, ground, (x, t) -> 0.0)],
        ),
        torso_grid,
    )

    torso_dh = torso_fun.dh
    torso_ch = torso_fun.ch

    transfer_op = NodalIntergridInterpolation(
        heart_dh,
        torso_dh,
        first(Ferrite.getfieldnames(heart_dh)),
        first(Ferrite.getfieldnames(torso_dh));
        subdomains_to = get_subdofhandler_indices_on_subdomains(torso_dh, torso_heart_domain),
    )

    source_op = setup_assembled_operator(
        strategy,
        BilinearDiffusionIntegrator(
            heart_diffusion_tensor_field,
            qrc,
            extracellular_potential_symbol,
        ),
        system_matrix_type,
        torso_dh,
    )
    update_operator!(source_op, nothing, TimeIntegrationContext(0.0, 0.0, 0.0)) # Trigger assembly

    torso_op = setup_assembled_operator(
        strategy,
        BilinearDiffusionIntegrator(
            torso_diffusion_tensor_field,
            qrc,
            extracellular_potential_symbol,
        ),
        system_matrix_type,
        torso_dh,
    )
    update_operator!(torso_op, nothing, TimeIntegrationContext(0.0, 0.0, 0.0)) # Trigger assembly

    # Setup electrodes
    ph = PointEvalHandler(torso_grid, electrode_positions; warn = false)
    if !all(x -> x !== nothing, ph.cells)
        error(
            "Poisson reconstruction setup failed! Some electrodes are not found in the torso mesh ($(ph.cells)).",
        )
    end

    PoissonECGReconstructionCache(
        heart_fun,
        torso_fun,
        source_op,
        torso_op,
        transfer_op,
        ph;
        linear_solver,
        solution_vector_type,
    )
end

function PoissonECGReconstructionCache(
    heart_fun::AffineODEFunction,
    torso_fun::AffineSteadyStateFunction,
    source_op::BilinearFerriteOperator,
    torso_op::BilinearFerriteOperator,
    transfer_op::AbstractTransferOperator,
    ph::PointEvalHandler;
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
)
    torso_dh = torso_op.engine.dh
    torso_ch = torso_fun.ch
    grid = get_grid(torso_dh)
    length(Ferrite.getfieldnames(torso_dh)) == 1 ||
        @warn "Multiple fields detected. Setup might be broken..."

    φₘt = create_system_vector(solution_vector_type, torso_fun) # RHS buffer for source term
    κ∇φₘt = create_system_vector(solution_vector_type, torso_fun) # RHS buffer after transfer
    κ∇φₘt .= 0.0
    ϕₑ = create_system_vector(solution_vector_type, torso_fun) # Solution vector

    linprob  = LinearSolve.LinearProblem(torso_op.A, κ∇φₘt; u0 = ϕₑ)
    lincache = init(linprob, linear_solver)

    return PoissonECGReconstructionCache(
        torso_op,
        source_op,
        transfer_op,
        ϕₑ,
        φₘt,
        κ∇φₘt,
        lincache,
        ph,
        torso_ch,
    )
end

function update_ecg!(cache::PoissonECGReconstructionCache, φₘ::AbstractVector)
    # Transfer φₘ to the torso
    transfer!(cache.φₘ_t, cache.transfer_op, φₘ)
    # Compute κᵢ∇φₘ on the torso
    mul!(cache.κ∇φₘ_t, cache.source_op, cache.φₘ_t)
    cache.κ∇φₘ_t[isnan.(cache.κ∇φₘ_t)] .= 0.0 # FIXME
    # "Move to right hand side
    cache.κ∇φₘ_t .*= -1.0
    # Apply BC to linear system
    apply_zero!(cache.inner_solver.A, cache.inner_solver.b, cache.ch)
    # Solve κ∇φₑ = -κᵢ∇φₘ for φₑ
    LinearSolve.solve!(cache.inner_solver)
    return nothing
end

# Batch evaluate all electrodes
function evaluate_ecg(cache::PoissonECGReconstructionCache)
    dh = cache.torso_op.engine.dh
    return evaluate_at_points(cache.ph, dh, cache.ϕₑ, first(dh.field_names))
end

"""
    Geselowitz1989ECGLeadCache(problem, torso_grid, κ, κᵢ, electrode_sets, [ground, linear_solver, solution_vector_type, system_matrix_type])

Here the lead field, `Z`, is computed using the discretization of `problem`.
The lead field is computed as the solution of
```math
\\nabla \\cdot(\\mathbf{\\kappa} \\nabla Z)=\\left\\{\\begin{array}{cl}
-1 & \\text { at the positive electrode } \\\\
1 & \\text { at the negative electrode } \\\\
0 & \\text { else where }
\\end{array}\\right.
```
Where ``\\kappa`` is the bulk conductivity tensor.

Returns a cache contain the lead fields that are used to compute the lead potentials as proposed in [Ges:1989:ote](@cite).
Calling [`reinit!`](@ref) with this method simply evaluates the following integral efficiently:

```math
V(t)=\\int \\nabla Z(\\boldsymbol{x}) \\cdot \\boldsymbol{\\kappa}_\\mathrm{i} \\nabla \\varphi_\\mathrm{m} \\,\\mathrm{d}\\boldsymbol{x}.
```
"""
struct Geselowitz1989ECGLeadCache{
    TZ <: AbstractMatrix,
    DiffusionOperatorType,
    TransferOperatorType,
    SolutionVectorType <: AbstractVector,
    ElectrodesVecType,
}
    source_op::DiffusionOperatorType   # Operator on the heart mesh for ∇κᵢ∇
    transfer_op::TransferOperatorType # Transfer from heart to torso mesh
    φₘ_t::SolutionVectorType          # Potential field on torso
    κ∇φₘ_t::SolutionVectorType        # Source term buffer on torso
    Z::TZ                             # Lead field
    electrode_positions::ElectrodesVecType
end

function Geselowitz1989ECGLeadCache(
    heart_fun::GenericSplitFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vector{<:Vec}};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    torso_heart_domain   = nothing,
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64, Int64},
)
    return Geselowitz1989ECGLeadCache(
        heart_fun.functions[1],
        torso_grid,
        heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
        full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
        [
            [get_closest_vertex(position, torso_grid) for position in positions] for
            positions in electrode_positions
        ];
        ipc,
        qrc,
        ground,
        torso_heart_domain,
        linear_solver,
        solution_vector_type,
        system_matrix_type,
    )
end

function Geselowitz1989ECGLeadCache(
    heart_fun::AffineODEFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vector{<:Vec}};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    torso_heart_domain   = nothing,
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64, Int64},
)
    return Geselowitz1989ECGLeadCache(
        heart_fun,
        torso_grid,
        heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
        full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
        [
            [get_closest_vertex(position, torso_grid) for position in positions] for
            positions in electrode_positions
        ];
        ipc,
        qrc,
        ground,
        torso_heart_domain,
        linear_solver,
        solution_vector_type,
        system_matrix_type,
    )
end

function Geselowitz1989ECGLeadCache(
    heart_fun::AffineODEFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid alone
    full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{Vector{VertexIndex}};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    torso_heart_domain   = nothing,
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64, Int64},
    lead_field_sym       = :Z,
    strategy             = SequentialAssemblyStrategy(PolyesterDevice()),
)
    tmpsym = heart_fun.bilinear_term.sym
    lead_field_model = SteadyDiffusionModel(
        full_diffusion_tensor_field,
        NoStimulationProtocol(), #ConstantCoefficient(NaN), # FIXME Poisoning to detecte if we accidentally touch these
        lead_field_sym,
    )

    source_model = SteadyDiffusionModel(
        heart_diffusion_tensor_field,
        NoStimulationProtocol(), #ConstantCoefficient(NaN), # FIXME Poisoning to detecte if we accidentally touch these
        tmpsym,
    )

    lead_field_fun = semidiscretize(
        Dict(name => lead_field_model for name in subdomain_names(torso_grid)),
        FiniteElementDiscretization(
            Dict(lead_field_sym => ipc);
            dbcs = [Dirichlet(lead_field_sym, ground, (x, t) -> 0.0)],
        ),
        torso_grid,
    )

    sourcefun = semidiscretize(
        Dict(name => source_model for name in subdomain_names(torso_grid)),
        FiniteElementDiscretization(Dict(tmpsym => ipc)),
        torso_grid,
    )

    ϕₘ_op = setup_assembled_operator(
        strategy,
        BilinearDiffusionIntegrator(heart_diffusion_tensor_field, qrc, tmpsym),
        system_matrix_type,
        sourcefun.dh,
    )
    update_operator!(ϕₘ_op, nothing, TimeIntegrationContext(0.0, 0.0, 0.0)) # Trigger assembly

    lead_op = setup_assembled_operator(
        strategy,
        BilinearDiffusionIntegrator(full_diffusion_tensor_field, qrc, lead_field_sym),
        system_matrix_type,
        lead_field_fun.dh,
    )
    update_operator!(lead_op, nothing, TimeIntegrationContext(0.0, 0.0, 0.0)) # Trigger assembly

    lead_field_dh = lead_field_fun.dh
    heart_dh = heart_fun.dh

    transfer_op = NodalIntergridInterpolation(
        heart_dh,
        lead_field_dh,
        first(Ferrite.getfieldnames(heart_dh)),
        first(Ferrite.getfieldnames(lead_field_dh));
        subdomains_to = get_subdofhandler_indices_on_subdomains(lead_field_dh, torso_heart_domain),
    )

    Geselowitz1989ECGLeadCache(
        heart_fun,
        lead_field_fun,
        lead_op,
        ϕₘ_op,
        transfer_op,
        electrode_positions;
        linear_solver,
        solution_vector_type,
        lead_field_sym,
    )
end

function Geselowitz1989ECGLeadCache(
    heart_fun::AffineODEFunction,
    lead_fun::AffineSteadyStateFunction,
    lead_op::BilinearFerriteOperator,
    source_op::BilinearFerriteOperator,
    transfer_op,
    electrode_positions::AbstractVector{Vector{VertexIndex}};
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    lead_field_sym       = :Z,
)
    lead_dh = lead_op.engine.dh
    length(Ferrite.getfieldnames(lead_dh)) == 1 ||
        @warn "Multiple fields detected. Setup might be broken..."
    nelectrodes = length(electrode_positions)
    φₘ_t        = create_system_vector(solution_vector_type, lead_fun) # Solution vector
    ∇φₘ_t       = create_system_vector(solution_vector_type, lead_fun)  # Solution vector
    Z           = zeros(eltype(∇φₘ_t), nelectrodes, length(∇φₘ_t))
    ϕₑ          = zeros(nelectrodes)

    lead_rhs = zeros(eltype(∇φₘ_t), nelectrodes, length(∇φₘ_t))

    leadprob = LinearSolve.LinearProblem(lead_op.A, copy(lead_rhs[1, :]))
    lincache = init(leadprob, linear_solver)
    @views for (i, electrode_set) in enumerate(electrode_positions)
        @assert length(electrode_set) ≥ 2 "Electrode set $i has too few electrodes ($(length(electrode_set))<2)"
        current_rhs = lead_rhs[i, :]
        _add_electrode!(current_rhs, lead_dh, electrode_set[1], 1.0, lead_field_sym)
        for j = 2:length(electrode_set)
            _add_electrode!(
                current_rhs,
                lead_dh,
                electrode_set[j],
                -1.0/(length(electrode_set)-1),
                lead_field_sym,
            )
        end
        lincache.b .= current_rhs
        LinearSolve.solve!(lincache)
        Z[i, :] .= lincache.u
    end

    return Geselowitz1989ECGLeadCache(source_op, transfer_op, φₘ_t, ∇φₘ_t, Z, electrode_positions)
end

function _add_electrode!(
    f::AbstractVector{T},
    dh::DofHandler,
    electrode::VertexIndex,
    weight,
    lead_field_sym::Symbol,
) where {T <: Number}
    local_dof = Ferrite.vertexdof_indices(
        Ferrite.getfieldinterpolation(dh.subdofhandlers[1], lead_field_sym),
    )[electrode[2]][1]::Int
    global_dof = celldofs(dh, electrode[1])[local_dof]::Int
    f[global_dof] = -weight
    return nothing
end

function update_ecg!(cache::Geselowitz1989ECGLeadCache, φₘ::AbstractVector)
    # Transfer κᵢ∇φₘ to the torso
    transfer!(cache.φₘ_t, cache.transfer_op, φₘ)
    # Compute κᵢ∇φₘ on the heart
    mul!(cache.κ∇φₘ_t, cache.source_op, cache.φₘ_t)
    cache.κ∇φₘ_t[isnan.(cache.κ∇φₘ_t)] .= 0.0 # FIXME
    return nothing
end

# Batch evaluate all electrodes
function evaluate_ecg(cache::Geselowitz1989ECGLeadCache)
    return -cache.Z * cache.κ∇φₘ_t
end
