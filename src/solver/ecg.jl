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

function _compute_quadrature_fluxes!(κ∇φₘ,dh,cv,φₘ,D)
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
    # TODO https://github.com/Ferrite-FEM/Ferrite.jl/pull/806 maybe?
    ip = dh.subdofhandlers[1].field_interpolations[1]
    # TODO QVector
    qr = op.element_cache.cellvalues.qr
    grid = Ferrite.get_grid(dh)
    κ∇φₘ = zeros(Ferrite.get_coordinate_type(grid), getnquadpoints(qr), getncells(dh.grid))
    cv = CellValues(qr, ip)
    _compute_quadrature_fluxes!(κ∇φₘ,dh,cv,φₘ,op.element_cache.integrator.D) # Function barrier
    Plonsey1964ECGGaussCache(κ∇φₘ, cv, dh.grid)
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
    @unpack κ∇φₘ, cv, grid = method
    for cell ∈ CellIterator(grid)
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
