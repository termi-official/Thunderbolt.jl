struct KellyEstimator
    p::Int
end
getdistance(p1::Vec{N, T},p2::Vec{N, T}) where {N, T} = norm(p1-p2);
getdiameter(cell_coords::AbstractVector{Vec{N, T}}) where {N, T} = maximum(getdistance.(cell_coords, reshape(cell_coords, (1,:))));

function estimate_kelly_interface!(T::Type{<:AbstractFloat}, err::AbstractVector, u::AbstractVector, interface_cache::InterfaceCache, interface_diffusion_cache::BilinearDiffusionInterfaceCache)
    error::T = 0.0
    facet_a = interface_cache.a.current_facet_id
    interface_coords = @view getcoordinates(interface_cache)[1][SVector(Ferrite.reference_facets(RefHexahedron)[facet_a])]
    h = getdiameter(interface_coords)
    p = 1
    @unpack interfacevalues, Dcache_here, Dcache_there = interface_diffusion_cache
    reinit!(interfacevalues, interface_cache)
    jump_term = 0
    for qp in QuadratureIterator(interfacevalues.here)
        D_here = evaluate_coefficient(Dcache_here, interface_cache, qp, time)
        D_there = evaluate_coefficient(Dcache_there, interface_cache, qp, time)
        normal = getnormal(interfacevalues, qp.i)
        Wf = getdetJdV(interfacevalues, qp.i) # maybe?
        ∇u_here = function_gradient(interfacevalues, qp.i, u; here = true)
        ∇u_there = function_gradient(interfacevalues, qp.i, u; here = false)
        jump_term += abs2(Wf*((D_here ⋅ ∇u_here ⋅ normal) - (D_there ⋅ ∇u_there ⋅ normal))) * Wf
    end
    ret = h/(2p)*jump_term
    err[cellid(interface_cache.a)] += ret
    err[cellid(interface_cache.b)] += ret
end
