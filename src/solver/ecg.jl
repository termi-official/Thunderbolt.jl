"""
    Plonsey1964ECGGaussCache(problem, κ)


Returns a cache to compute the lead field with the form proposed in [Plo:1964:vcf](@cite)
with the Gauss theorem applied to it, as for example described in [OgiBalPer:2021:ema](@cite).
Calling [`evaluate_ecg`](@ref) with this method simply evaluates the following integral efficiently:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_\\mathrm{t}} \\int_\\Omega \\frac{ \\kappa_ \\nabla \\varphi_\\mathrm{m} \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

Where φₘ is the solution vector containing the transmembranepotential, and κₜ is the torso's conductivity.

The important simplifications taken are:
   1. Surrounding volume is an infinite, homogeneous sphere with isotropic conductivity
   2. The extracellular space and surrounding volume share the same isotropic, homogeneous conductivity tensor
"""
struct Plonsey1964ECGGaussCache{BUF, CV, DH <: DofHandler, κT}
    # Buffer for storing "κ(x) ∇φₘ(x,t)" at the quadrature points
    κ∇φₘ::BUF
    dh::DH
    cv::CV
    κ::κT
end

function Plonsey1964ECGGaussCache(problem::SplitProblem{<: TransientHeatProblem}, κ)
    heat_problem = problem.A
    dh_ϕₘ = heat_problem.dh
    @assert length(dh_ϕₘ.subdofhandlers) == 1 "TODO subdomain support"
    @assert length(dh_ϕₘ.subdofhandlers[1].field_interpolations) == 1 "Problem setup might be broken..."
    sdim = Ferrite.getdim(dh_ϕₘ.grid)

    ip = dh_ϕₘ.subdofhandlers[1].field_interpolations[1]
    qr = QuadratureRule{getrefshape(ip)}(2*Ferrite.getorder(ip)) # TODO Mabe make this part of the problem or create a wrapper for it with qr?
    cv = CellValues(qr, ip)
    
    κ∇φₘ = zeros(Vec{sdim}, getnquadpoints(qr), getncells(dh_ϕₘ.grid))

    return Plonsey1964ECGGaussCache(κ∇φₘ, dh_ϕₘ, cv, κ)
end

function Ferrite.reinit!(cache::Plonsey1964ECGGaussCache, φₘ)
    @unpack κ∇φₘ, dh, cv, κ = cache
    _compute_quadrature_fluxes!(κ∇φₘ,dh,cv,φₘ,κ) # Function barrier
end

"""
    Geselowitz1989ECGLeadCache(problem, κ, κᵢ, electordes, electrode_pairs, [zero_vertex])

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
struct Geselowitz1989ECGLeadCache{T, D, ∇φₘT, ZT <: AbstractArray{T}, ∇ZT <: AbstractArray{Vec{D, T}}, OP <: AssembledBilinearOperator, PreconditionerT, V, κᵢT}
    ∇Z::∇ZT
    κ∇φₘ::∇φₘT
    op::OP
    p::PreconditionerT
    v::V
    κᵢ::κᵢT
    # Does is make sense to store these?
    Z::ZT
    zero_vertex::VertexIndex
    electrodes::Vector{Vec{3, T}}
    electrode_pairs::Vector{Tuple{Int, Int}}
end

function Geselowitz1989ECGLeadCache(problem::SplitProblem{<:TransientHeatProblem}, κ, κᵢ,
    electrodes::Vector{Vec{3, T}}, electrode_pairs::Vector{Tuple{Int, Int}}, zero_vertex::VertexIndex = VertexIndex(1,1)) where T
    heat_problem = problem.A
    dh_ϕₘ = heat_problem.dh
    grid = dh_ϕₘ.grid
    @assert length(dh_ϕₘ.subdofhandlers) == 1 "TODO subdomain support"
    @assert length(dh_ϕₘ.subdofhandlers[1].field_interpolations) == 1 "Problem setup might be broken..."
    sdim = Ferrite.getdim(grid)

    nleadfields = length(electrode_pairs)

    ip = dh_ϕₘ.subdofhandlers[1].field_interpolations[1]
    qr = QuadratureRule{getrefshape(ip)}(2*Ferrite.getorder(ip)) # TODO Mabe make this part of the problem or create a wrapper for it with qr?
    cv = CellValues(qr, ip)

    dh_Z = DofHandler(grid)
    Ferrite.add!(dh_Z, :Z, ip)
    close!(dh_Z);
    κ∇φₘ = zeros(Vec{sdim}, getnquadpoints(qr), getncells(grid))
    Z = Matrix{T}(undef, nleadfields, ndofs(dh_Z))
    ∇Z = zeros(Vec{sdim, T}, nleadfields, getnquadpoints(qr), getncells(grid))
    K = create_sparsity_pattern(dh_Z)
    f = zeros(ndofs(dh_Z))
    v = zeros(size(Z,1))

    integrator = BilinearDiffusionIntegrator(-κ)
    element_cache = BilinearDiffusionElementCache(integrator, cv)
    op = AssembledBilinearOperator(K, element_cache, dh_Z)
    update_operator!(op, 0.)

    ch = ConstraintHandler(dh_Z)
    Ferrite.add!(ch, Dirichlet(:Z, Set([zero_vertex]), (x, t) -> 0.0))
    close!(ch);
    apply!(op.A, f, ch)
    ml = ruge_stuben(op.A)
    p = aspreconditioner(ml)
    for (i, electrode_pair) in enumerate(electrode_pairs)
        fill!(f, zero(T))
        Z_lead = @view Z[i, :]
        ∇Z_lead = @view ∇Z[i, :, :]
        _compute_lead_field(f, ∇Z_lead, Z_lead, op, p, electrodes[[electrode_pair[1], electrode_pair[2]]])
    end
    
    return Geselowitz1989ECGLeadCache(∇Z, κ∇φₘ, op, p, v, κᵢ, Z, zero_vertex, electrodes, electrode_pairs)
end

function _compute_lead_field(f, ∇Z, Z, op, p, electrodes::AbstractArray{Vec{3, T}}) where T
    _add_electrode(f, op.dh, electrodes[1], true)
    _add_electrode(f, op.dh, electrodes[2], false)
    (temp, _) = Krylov.cg(op.A,f, M=p, ldiv = true)
    Z .= temp
    _compute_quadrature_fluxes!(∇Z,op.dh,op.element_cache.cellvalues,Z,ConstantCoefficient(SymmetricTensor{2,3}((
        1, 0, 0,
           1, 0,
              1
    ))))
end

function _add_electrode(f::AbstractVector{T}, dh::DofHandler, electrode::Vec{3,T}, is_positive::Bool) where T<:Number
    grid = dh.grid
    haskey(grid.vertexsets, "$electrode") || addvertexset!(grid, "$electrode", x -> x ≈ electrode)
    vertex = first(getvertexset(grid, "$electrode"))
    local_dof = Ferrite.vertexdof_indices(Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :Z))[vertex[2]][1]::Int
    global_dof = celldofs(dh, vertex[1])[local_dof]::Int
    f[global_dof] = is_positive ? -1 : 1
    return nothing
end

"""
    reinit!(cache::Geselowitz1989ECGLeadCache, ϕₘ)

Compute the potential between lead pairs by evaluating:

```math
V(t)=\\int \\nabla Z(\\boldsymbol{x}) \\cdot \\boldsymbol{\\kappa}_\\mathrm{i} \\nabla \\varphi_\\mathrm{m} \\,\\mathrm{d}\\boldsymbol{x}.
```
"""
function Ferrite.reinit!(cache::Geselowitz1989ECGLeadCache, ϕₘ)
    @unpack κ∇φₘ, ∇Z, op, v, κᵢ = cache
    cv = op.element_cache.cellvalues
    dh = op.dh
    _compute_quadrature_fluxes!(κ∇φₘ, op.dh, cv, ϕₘ, κᵢ)
    fill!(v, zero(eltype(v)))
    for cell ∈ CellIterator(dh)
        reinit!(cv, cell)
        κ∇φₘe = @view κ∇φₘ[:, cellid(cell)]
        ∇Ze = @view ∇Z[:, :, cellid(cell)]
        v .+= _evaluate_ecg_geselowitz(κ∇φₘe, ∇Ze, cv)
    end
end

"""
    Potse2006ECGPoissonReconstructionCache(mesh, κ, qr_collection, ip_collection, zero_vertex::VertexIndex)
    
Sets up a cache for calculating ``\\varphi_\\mathrm{e}`` by solving the Poisson problem
```math
\\nabla \\cdot \\boldsymbol{\\kappa} \\nabla \\varphi_{\\mathrm{e}}=-\\nabla \\cdot\\left(\\boldsymbol{\\kappa}_{\\mathrm{i}} \\nabla \\varphi_\\mathrm{m}\\right)
```
as proposed in [PotDubRicVinGul:2006:cmb](@cite) and mentioned in [OgiBalPer:2021:ema](@cite). Where κ is the bulk conductivity tensor, and κᵢ is the intracellular conductivity tensor. The cache includes the assembled 
stiffness matrix with applied homogeneous Dirichlet boundary condition at the first vertex of the mesh. As the problem is solved for each timestep with only the right hand side changing.

TODO: Implement [BisPla:2011:bes](@cite) to improve the precision.
"""
struct Potse2006ECGPoissonReconstructionCache{ϕT <: AbstractVector, OP, PreconditionerT, FT <: AbstractVector, κT}
    ϕₑ::ϕT
    op::OP
    p::PreconditionerT
    f::FT
    κᵢ::κT
end

function Potse2006ECGPoissonReconstructionCache(problem::SplitProblem{<: TransientHeatProblem}, κ,
    κᵢ, zero_vertex::VertexIndex = VertexIndex(1,1))

    heat_problem = problem.A
    dh_ϕₘ = heat_problem.dh
    grid = dh_ϕₘ.grid
    @assert length(dh_ϕₘ.subdofhandlers) == 1 "TODO subdomain support"
    @assert length(dh_ϕₘ.subdofhandlers[1].field_interpolations) == 1 "Problem setup might be broken..."
    
    ip = dh_ϕₘ.subdofhandlers[1].field_interpolations[1]
    qr = QuadratureRule{getrefshape(ip)}(2*Ferrite.getorder(ip)) # TODO Mabe make this part of the problem or create a wrapper for it with qr?
    cv = CellValues(qr, ip)
    dh_ϕₑ = DofHandler(grid)
    Ferrite.add!(dh_ϕₑ, :ϕₑ, ip)
    close!(dh_ϕₑ);

    K = create_sparsity_pattern(dh_ϕₑ)
    ϕₑ = zeros(Ferrite.ndofs(dh_ϕₑ))
    f = similar(ϕₑ)
    integrator = BilinearDiffusionIntegrator(-κ)
    element_cache = BilinearDiffusionElementCache(integrator, cv)
    op = AssembledBilinearOperator(K, element_cache, dh_ϕₑ)
    update_operator!(op, 0.)

    ch = ConstraintHandler(dh_ϕₑ)
    Ferrite.add!(ch, Dirichlet(:ϕₑ, Set([zero_vertex]), (x, t) -> 0.0))
    close!(ch);
    apply!(op.A, f, ch)
    p = aspreconditioner(ruge_stuben(op.A))

    return Potse2006ECGPoissonReconstructionCache(ϕₑ, op, p, f, κᵢ)
end

"""
    Ferrite.reinit!(cache::Potse2006ECGPoissonReconstructionCache, ϕₘ)

Compute ``\\varphi_\\mathrm{e}`` by solving the equation:

```math
\\nabla \\cdot \\boldsymbol{\\kappa} \\nabla \\phi_{\\mathrm{e}}=-\\nabla \\cdot\\left(\\boldsymbol{\\kappa}_{\\mathrm{i}} \\nabla \\varphi_\\mathrm{m}\\right)
```
for the given ``\\varphi_m``
"""
function Ferrite.reinit!(cache::Potse2006ECGPoissonReconstructionCache, ϕₘ)
    @unpack ϕₑ, op, p, f, κᵢ = cache
    cv = op.element_cache.cellvalues
    dh = op.dh
    n_basefuncs = getnbasefunctions(cv)
    fill!(f, 0.0)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
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
    ch = ConstraintHandler(dh)
    Ferrite.add!(ch, Dirichlet(:ϕₑ, Set([VertexIndex(1,1)]), (x, t) -> 0.0))
    close!(ch);
    apply_zero!(op.A, f, ch)
    (temp, _) = Krylov.cg(op.A,f, M=p, ldiv = true)
    ϕₑ .= temp
end

"""
    evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)

Compute the pseudo ECG at a given point x by evaluating:

\$\\varphi_\\mathrm{e}(x)=\\frac{1}{4 \\pi \\kappa_\\mathrm{t}} \\int_\\Omega \\frac{ \\kappa_ \\nabla \\varphi_\\mathrm{m} \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

For more information please read the docstring for [`Plonsey1964ECGGaussCache`](@ref)
"""
function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)
    φₑ = 0.0
    @unpack κ∇φₘ, cv, dh = method
    for cell ∈ CellIterator(dh.grid)
        reinit!(cv, cell)
        coords = getcoordinates(cell)
        κ∇φₘe = @view κ∇φₘ[:, cellid(cell)]
        φₑ += _evaluate_ecg_plonsey_gauss(κ∇φₘe, coords, cv, x)
    end

    return -φₑ / (4π*κₜ)
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

"""
    evaluate_ecg(method::Geselowitz1989ECGLeadCache, lead_index::Int)

Returns the potential between the lead pair associated with the index `lead_index` for a [`reinit!`](@ref)ed cache.

For more information please read the docstring for [`Geselowitz1989ECGLeadCache`](@ref)
"""
function evaluate_ecg(method::Geselowitz1989ECGLeadCache, lead_index::Int)
    return method.v[lead_index]
end
function evaluate_ecg(method::Geselowitz1989ECGLeadCache, x::NTuple{2, <:Vec})
    @unpack ∇Z, κ∇φₘ, op, electrodes, v, electrode_pairs = method
    p₁ = findfirst(_x -> _x ≈ x[1], electrodes)
    p₂ = findfirst(_x -> _x ≈ x[2], electrodes)    
    i = findfirst(_x -> _x ∈ ((p₁, p₂),(p₂, p₁)), electrode_pairs)
    any(isnothing.((p₁, p₂, i))) && throw(ArgumentError("There exists no electrode pairs with the provided coordinates"))
    return v[i]
end

function _evaluate_ecg_geselowitz(κ∇φₘ, ∇Z, cv)
    V_local = zeros(size(∇Z,1))
    for qp in QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)
        for (i, _Z) in enumerate(V_local)
            V_local[i] += ∇Z[i, qp.i] ⋅ κ∇φₘ[qp.i] * dΩ 
        end
    end
    return V_local
end

"""
    evaluate_ecg(method::Potse2006ECGPoissonReconstructionCache, x::Vec)

Evaluate ``\\varphi_\\mathrm{e}`` at the point `x` using `Ferrite.PointEvalHandler`.

For more information please read the docstring for [`Potse2006ECGPoissonReconstructionCache`](@ref)
"""
function evaluate_ecg(method::Potse2006ECGPoissonReconstructionCache, x::Vec)
    @unpack op, ϕₑ = method
    dh = op.dh
    ph = PointEvalHandler(dh.grid, [x]) # Cache this?
    ϕₑ = evaluate_at_points(ph, dh, ϕₑ, :ϕₑ)[1]
    return ϕₑ
end

function _compute_quadrature_fluxes!(κ∇φₘ::AbstractArray{T},dh,cv,φₘ,D) where T
    fill!(κ∇φₘ, zero(T))
    for cell ∈ CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        φₘₑ = @view φₘ[celldofs(cell)]

        for qp in QuadratureIterator(cv)
            D_loc = evaluate_coefficient(D, cell, qp, time)
            for i in 1:getnbasefunctions(cv)
                ∇Nᵢ = shape_gradient(cv, qp, i)
                κ∇φₘ[qp.i, cellid(cell)] += D_loc ⋅ ∇Nᵢ * φₘₑ[i]
            end
        end
    end
end
