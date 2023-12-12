"""
    evaluate_ecg(method, x::Vec)

Evaluate the ECG with a given method at a single point.
"""
evaluate_ecg(method, x::Vec)

"""
    Plonsey1964ECGGaussCache(problem, op)

Cache to compute the lead field with the form proposed in [Plo:1964:vcf](@cite)
with the Gauss theorem applied to it, as for example described in [OgiBalPer:2021:](@cite):
TODO formula

TODO citations
    * Original formulation  https://www.sciencedirect.com/science/article/pii/S0006349564867850?via%3Dihub
    * Numerical treatment  https://link.springer.com/chapter/10.1007/978-3-030-78710-3_48
TODO who proposed this one first?
"""
struct Plonsey1964ECGGaussCache{BUF, CV, G, T}
    # Buffer for storing "κ(x) ∇φₘ(x,t)" at the quadrature points
    κ∇φₘ::BUF
    cv::CV
    grid::G
    κₜ::T
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
                κ∇φₘ[qp.i, cellid(cell)] = D_loc ⋅ ∇Nᵢ * φₘₑ[i]
            end
        end
    end
end

# TODO better abstraction layer
function Plonsey1964ECGGaussCache(problem::SplitProblem{<:TransientHeatProblem}, op::AssembledBilinearOperator, φₘ, κₜ)
    @unpack dh = problem.A
    @assert length(dh.subdofhandlers) == 1 "TODO subdomain support"
    @assert length(dh.subdofhandlers[1].field_interpolations) == 1 "Problem setup might be broken..."
    sdim = Ferrite.getdim(dh.grid)
    # TODO https://github.com/Ferrite-FEM/Ferrite.jl/pull/806 maybe?
    ip = dh.subdofhandlers[1].field_interpolations[1]
    qr = QuadratureRule{Ferrite.getrefshape(ip)}(2*Ferrite.getorder(ip))
    κ∇φₘ = zeros(Vec{sdim}, getnquadpoints(qr), getncells(dh.grid))
    cv = CellValues(qr, ip)
    _compute_quadrature_fluxes!(κ∇φₘ,dh,cv,φₘ,op.element_cache.integrator.D) # Function barrier
    Plonsey1964ECGGaussCache(κ∇φₘ, cv, dh.grid, κₜ)
end

function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec)
    φₑ = 0.0
    @unpack κ∇φₘ, κₜ, cv, grid = method
    for cell ∈ CellIterator(grid.grid)
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
        fecv_J = zero(Tensor{2,dim,T})
        @inbounds for j in 1:n_geom_basefuncs
            fecv_J += coords[j] ⊗ cv.dMdξ[j, qp]
        end
        detJ = det(fecv_J)
        dΩ = detJ * w
        # Compute x̃
        x̃ = zero(Vec{dim,T})
        @inbounds for j in 1:n_geom_basefuncs
            x̃ += Ferrite.geometric_value(cv, qp, j) * coords[j]
        end
        # Evaluate κ∇φₘ*(x̃-x)/||x̃-x||³
        φₑ_local += κ∇φₘ[qp] ⋅ (x̃-x)/norm((x̃-x))^3 * dΩ
    end
    return φₑ_local
end
