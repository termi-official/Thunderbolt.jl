"""
    Plonsey1964ECGGaussCache(problem, op, φₘ)

Here φₘ is the solution vector containing the transmembranepotential, op is the associated diffusion opeartor and 
κₜ is the torso's conductivity.

Returns a cache to compute the lead field with the form proposed in [Plo:1964:vcf](@cite)
with the Gauss theorem applied to it, as for example described in [OgiBalPer:2021:ema](@cite).
Calling [`evaluate_ecg`](@ref) with this method simply evaluates the following integral efficiently:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_t} \\int_\\Omega \\frac{ \\kappa_ ∇φₘ \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

The important simplifications taken are:
   1. Surrounding volume is an infinite, homogeneous sphere with isotropic conductivity
   2. The extracellular space and surrounding volume share the same isotropic, homogeneous conductivity tensor
"""
struct Plonsey1964ECGGaussCache{BUF, CV, G}
    # Buffer for storing "κ(x) ∇φₘ(x,t)" at the quadrature points
    κ∇φₘ::BUF
    cv::CV
    grid::G
end

struct LeadFieldCache{T, D, ZT <: AbstractArray{T}, ∇φT <: AbstractArray{Vec{D, T}}, DH <: DofHandler, CV <: CellValues}
    Z::ZT
    κ∇φₘ::∇φT
    dh::DH
    cv::CV
    # Does is make sense to store these?
    zero_vertex::VertexIndex
    electrodes::Vector{Vec{3, T}}
    electrode_pairs::Vector{Tuple{Int, Int}}
end

struct ECG_ReconstructionCache{ϕT <: AbstractVector, DH <: AbstractDofHandler, CV <: AbstractCellValues, KT <: AbstractMatrix, FT <: AbstractVector}
    ϕₑ::ϕT
    dh::DH
    cv::CV
    K::KT
    f::FT
end

function ecg_reconstruction_cache(mesh,
    κ::SpectralTensorCoefficient,
    qr_collection::QuadratureRuleCollection,
    ip_collection::ScalarInterpolationCollection,
    zero_vertex::VertexIndex,
    )
    ip = getinterpolation(ip_collection, getcells(mesh,1))
    qr = getquadraturerule(qr_collection, getcells(mesh,1))
    cv = CellValues(qr, ip);

    dh = DofHandler(mesh)
    Ferrite.add!(dh, :Z, ip)
    close!(dh);
    K = create_sparsity_pattern(dh)
    ϕₑ = zeros(Ferrite.ndofs(dh))
    f = similar(ϕₑ)
    integrator = BilinearDiffusionIntegrator(κ)
    element_cache = BilinearDiffusionElementCache(integrator, cv)
    op = AssembledBilinearOperator(K, element_cache, dh)
    update_operator!(op, 0.)

    ch = ConstraintHandler(dh)
    Ferrite.add!(ch, Dirichlet(:Z, Set([zero_vertex]), (x, t) -> 0.0))
    close!(ch);
    apply!(op.A, f, ch)

    return ECG_ReconstructionCache(ϕₑ, dh, cv, K, f)
end

function _add_electrodes(f::Vector{T}, dh::DofHandler, set_name::String, positive_electrode::Vec{3,T}, negative_electrode::Vec{3,T}) where T<:Number
    mesh = dh.grid
    haskey(mesh.vertexsets, set_name) && throw(ArgumentError("Electrode vertexset name $(set_name) already exists"))
    addvertexset!(mesh, set_name*"-p", x -> x ≈ positive_electrode)
    addvertexset!(mesh, set_name*"-n", x -> x ≈ negative_electrode)
    positive_vertex = first(getvertexset(mesh, set_name*"-p"))
    negative_vertex = first(getvertexset(mesh, set_name*"-n"))

    local_positive_dof = Ferrite.vertexdof_indices(Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :Z))[positive_vertex[2]][1]::Int
    global_positive_dof = celldofs(dh, positive_vertex[1])[local_positive_dof]::Int
    f[global_positive_dof] = -1
    local_negative_dof = Ferrite.vertexdof_indices(Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :Z))[negative_vertex[2]][1]::Int
    global_negative_dof = celldofs(dh, negative_vertex[1])[local_negative_dof]::Int
    f[global_negative_dof] = 1
end

function compute_lead_field(mesh, _κ::SpectralTensorCoefficient,
    qr_collection::QuadratureRuleCollection,
    ip_collection::ScalarInterpolationCollection,
    zero_vertex::VertexIndex,
    electrodes::Vector{Vec{3, T}},
    electrode_pairs::Vector{Tuple{Int, Int}}) where T
    sdim = Ferrite.getdim(mesh)
    ets = elementtypes(mesh)
    @assert length(ets) == 1 "Multiple elements not supported yet." 

    nleadfields = length(electrode_pairs)

    ip = getinterpolation(ip_collection, getcells(mesh,1))
    qr = getquadraturerule(qr_collection, getcells(mesh,1))
    cv = CellValues(qr, ip);

    dh = DofHandler(mesh)
    Ferrite.add!(dh, :Z, ip)
    close!(dh);
    Z = Matrix{T}(undef, nleadfields, ndofs(dh))
    κ∇φₘ = zeros(Vec{sdim, T}, getnquadpoints(qr), getncells(dh.grid))
    lead_field = LeadFieldCache(Z, κ∇φₘ, dh, cv, zero_vertex, electrodes, electrode_pairs)
    _compute_lead_field(lead_field, _κ)
    return lead_field
end

function _compute_lead_field(lead_field::LeadFieldCache, _κ::SpectralTensorCoefficient)
    
    @unpack Z, dh, cv, zero_vertex, electrodes, electrode_pairs = lead_field

    K = create_sparsity_pattern(dh)
    f = zeros(ndofs(dh))

    integrator = BilinearDiffusionIntegrator(_κ)
    element_cache = BilinearDiffusionElementCache(integrator, cv)
    op = AssembledBilinearOperator(K, element_cache, dh)
    update_operator!(op, 0.)

    ch = ConstraintHandler(dh)
    Ferrite.add!(ch, Dirichlet(:Z, Set([zero_vertex]), (x, t) -> 0.0))
    close!(ch);
    apply!(op.A, f, ch)

    for (i, electrode_pair) in enumerate(electrode_pairs)
        _add_electrodes(f, dh, "lead_field_$(i)", electrodes[electrode_pair[1]], electrodes[electrode_pair[2]])
        ml = ruge_stuben(op.A)
        p = aspreconditioner(ml)
        Z_lead = @view Z[i, :]
        IterativeSolvers.cg!(Z_lead, op.A, f, Pl=p) #8710 - 116ms
        # Krylov.cg(op.A, f, M=p, ldiv=true) #errors
        # Krylov.cgs(op.A, f, M=p, ldiv=true) #8718 - 161ms
        # Krylov.gmres(op.A, f, M=p, ldiv=true) #8776 - 120ms
    end    
end

function reconstruct_ecg(cache::ECG_ReconstructionCache, κᵢ, ϕₘ)
    
    @unpack ϕₑ, dh, cv, K, f = cache

    n_basefuncs = getnbasefunctions(cv)
    fill!(f, 0.0)
    for (cell_num, cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
        φₘe = @view ϕₘ[celldofs(cell)]
        fₑ = @view f[celldofs(cell)]

        for qp in QuadratureIterator(cv)
            dΩ = getdetJdV(cv, qp)
            κᵢ_val = evaluate_coefficient(κᵢ, cell, qp, 0.0)
            ∇ϕₘ = function_gradient(cv, qp, φₘe)
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cv, qp, j)
                fₑ[j] -= ∇Nⱼ ⋅ (κᵢ_val ⋅ ∇ϕₘ) * dΩ
            end
        end
    end
    # op.A\f
    ml = ruge_stuben(K)
    p = aspreconditioner(ml)
    IterativeSolvers.cg!(ϕₑ, K, f, Pl=p)
end

function _compute_quadrature_fluxes!(κ∇φₘ::AbstractArray{T},dh,cv,φₘ,D) where T
    fill!(κ∇φₘ, zero(T))
    for cell ∈ CellIterator(dh)
        n_basefuncs = getnbasefunctions(cv)

        reinit!(cv, cell)
        φₘₑ = @view φₘ[celldofs(cell)]

        for qp in QuadratureIterator(cv)
            D_loc = evaluate_coefficient(D, cell, qp, time)
            # dΩ = getdetJdV(cellvalues, qp)
            for i in 1:n_basefuncs
                ∇Nᵢ = shape_gradient(cv, qp, i)
                κ∇φₘ[qp.i, cellid(cell)] += D_loc ⋅ ∇Nᵢ * φₘₑ[i]
            end
        end
    end
end

# TODO better abstraction layer
function Plonsey1964ECGGaussCache(dh::DofHandler, op::AssembledBilinearOperator, φₘ)
    @assert length(dh.subdofhandlers) == 1 "TODO subdomain support"
    @assert length(dh.subdofhandlers[1].field_interpolations) == 1 "Problem setup might be broken..."
    sdim = Ferrite.getdim(dh.grid)
    # TODO https://github.com/Ferrite-FEM/Ferrite.jl/pull/806 maybe?
    ip = dh.subdofhandlers[1].field_interpolations[1]
    # TODO QVector
    qr = op.element_cache.cellvalues.qr
    κ∇φₘ = zeros(Vec{sdim}, getnquadpoints(qr), getncells(dh.grid))
    cv = CellValues(qr, ip)
    _compute_quadrature_fluxes!(κ∇φₘ,dh,cv,φₘ,op.element_cache.integrator.D) # Function barrier
    Plonsey1964ECGGaussCache(κ∇φₘ, cv, dh.grid)
end

function Plonsey1964ECGGaussCache(problem::SplitProblem{<:TransientHeatProblem}, op::AssembledBilinearOperator, φₘ)
    @unpack dh = problem.A
    Plonsey1964ECGGaussCache(dh, op, φₘ)
end

"""
    evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)

Compute the pseudo ECG at a given point x by evaluating:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_t} \\int_\\Omega \\frac{ \\kappa_ ∇φₘ \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

For more information please read the docstring for [`Plonsey1964ECGGaussCache`](@ref)
"""
function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)
    φₑ = 0.0
    @unpack κ∇φₘ, cv, grid = method
    for cell ∈ CellIterator(grid)
        reinit!(cv, cell)
        coords = getcoordinates(cell)
        κ∇φₘe = @view κ∇φₘ[:, cellid(cell)]
        φₑ += _evaluate_ecg_plonsey_gauss(κ∇φₘe, coords, cv, x)
    end

    return -φₑ / (4π*κₜ)
end

function evaluate_ecg(method::LeadFieldCache, φₘ::Vector, κ::SpectralTensorCoefficient)
    @unpack Z, κ∇φₘ, cv, dh = method
    V = zeros(size(Z,1))
    _compute_quadrature_fluxes!(κ∇φₘ, dh, cv, φₘ, κ)
    for cell ∈ CellIterator(dh)
        reinit!(cv, cell)
        κ∇φₘe = @view κ∇φₘ[:, cellid(cell)]
        Ze = @view Z[:, celldofs(cell)]
        V .+= _evaluate_ecg_lead(κ∇φₘe, Ze, cv)
    end

    return V
end

function _evaluate_ecg_plonsey_gauss(κ∇φₘ, coords::AbstractVector{Vec{dim,T}}, cv, x::Vec{dim,T}) where {dim, T}
    φₑ_local = 0.0
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    @inbounds for (qp, w) in pairs(Ferrite.getweights(cv.qr))
        # Compute dΩ
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, qp, coords)
        dΩ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping)) * w
        # Compute x̃
        x̃ = spatial_coordinate(cv, qp, coords)
        # Evaluate κ∇φₘ*(x̃-x)/||x̃-x||³
        φₑ_local += κ∇φₘ[qp] ⋅ (x̃-x)/norm((x̃-x))^3 * dΩ
    end
    return φₑ_local
end

function _evaluate_ecg_lead(κ∇φₘ, Z, cv)
    V_local = zeros(size(Z,1))
    for qp in QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)
        for (i, _Z) in enumerate(eachrow(Z))
            ∇Z = function_gradient(cv, qp, _Z)
            V_local[i] += ∇Z ⋅ κ∇φₘ[qp.i] * dΩ 
        end
    end
    return V_local
end
