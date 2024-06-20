
# TODO where to put this?
function construct_qvector(::Type{StorageType}, ::Type{IndexType}, mesh::SimpleMesh, qrc::QuadratureRuleCollection, subdomains::Vector{String} = [""]) where {StorageType, IndexType}
    num_points = 0
    num_cells  = 0
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
            num_points += getnquadpoints(qr)*length(cellset)
            num_cells  += length(cellset)
        end
    end
    data    = zeros(eltype(StorageType), num_points)
    offsets = zeros(num_cells+1)

    offsets[1]        = 1
    next_point_offset = 1
    next_cell         = 1
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
            for cellidx in cellset
                next_point_offset += getnquadpoints(qr)
                next_cell += 1
                offsets[next_cell] = next_point_offset
            end
        end
    end
    
    return DenseDataRange(StorageType(data), IndexType(offsets))
end

function compute_quadrature_fluxes!(fluxdata, op, u, field_name)
    grid = get_grid(op.dh)
    sdim = getspatialdim(grid)
    for sdh in op.dh.subdofhandlers
        ip          = Ferrite.getfieldinterpolation(sdh, field_name)
        firstcell   = getcells(grid, first(sdh.cellset))
        ip_geo      = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
        element_qr  = getquadraturerule(op.element_qrc, firstcell)
        cv = CellValues(element_qr, ip, ip_geo)
        _compute_quadrature_fluxes_on_subdomain!(fluxdata,sdh,cv,u,op)
    end
end

function _compute_quadrature_fluxes_on_subdomain!(fluxdata,sdh,cv,u,op::AbstractBilinearOperator)
    return _compute_quadrature_fluxes_on_subdomain!(fluxdata,sdh,cv,u,op.integrator)
end

function _compute_quadrature_fluxes_on_subdomain!(κ∇u,sdh,cv,u,integrator::BilinearDiffusionIntegrator)
    n_basefuncs = getnbasefunctions(cv)
    for cell ∈ CellIterator(sdh)
        κ∇ucell = get_data_for_index(κ∇u, cellid(cell))

        reinit!(cv, cell)
        uₑ = @view u[celldofs(cell)]

        for qp in QuadratureIterator(cv)
            D_loc = evaluate_coefficient(integrator.D, cell, qp, time)
            # dΩ = getdetJdV(cellvalues, qp)
            for i in 1:n_basefuncs
                ∇Nᵢ = shape_gradient(cv, qp, i)
                κ∇ucell[qp.i] += D_loc ⋅ ∇Nᵢ * uₑ[i]
            end
        end
    end
end

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
struct Plonsey1964ECGGaussCache{BufferType, OperatorType}
    # Buffer for storing "κ(x) ∇φₘ(x,t)" at the quadrature points
    κ∇φₘ::BufferType
    op::OperatorType
end

function Plonsey1964ECGGaussCache(dh::DofHandler{sdim}, op::AssembledBilinearOperator, φₘ) where sdim
    @assert length(op.dh.field_names) == 1 "Problem setup might be broken..."
    grid = get_grid(dh)
    # FIXME we should grab the subdomain information somehow from op
    κ∇φₘ = construct_qvector(Vector{Vec{sdim,Float64}}, Vector{Int64}, grid, op.element_qrc)
    compute_quadrature_fluxes!(κ∇φₘ, op, φₘ, op.dh.field_names[1])
    Plonsey1964ECGGaussCache(κ∇φₘ, op)
end

function Plonsey1964ECGGaussCache(problem::OperatorSplittingProblem, op::AssembledBilinearOperator, φₘ)
    @unpack dh = get_operator(problem.f, 1)
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
    @unpack κ∇φₘ, op = method
    @unpack dh = op
    @assert length(dh.field_names) == 1 "Problem setup might be broken..."
    grid = get_grid(dh)
    sdim = getspatialdim(grid)
    for sdh in dh.subdofhandlers
        ip          = Ferrite.getfieldinterpolation(sdh, first(dh.field_names))
        firstcell   = getcells(grid, first(sdh.cellset))
        ip_geo      = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
        element_qr  = getquadraturerule(op.element_qrc, firstcell)
        cv = CellValues(element_qr, ip, ip_geo)
        # Function barrier
        φₑ += _evaluate_ecg_inner!(κ∇φₘ, method, x, κₜ, sdh, cv)
    end

    return -φₑ / (4π*κₜ)
end

function _evaluate_ecg_inner!(κ∇φₘ, method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real, sdh, cv)
    φₑ = 0.0
    for cell ∈ CellIterator(sdh)
        reinit!(cv, cell)
        coords = getcoordinates(cell)
        κ∇φₘe = get_data_for_index(κ∇φₘ, cellid(cell))
        φₑ += _evaluate_ecg_plonsey_gauss(κ∇φₘe, coords, cv, x)
    end
    return φₑ
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
