struct InternalVariableInfo
    name::Symbol
    size::Int
end

# struct SubLocalVariableHandler
#     names::Vector{Symbol} # Contains symbols for all variables in the handler
#     local_ranges::Vector{UnitRange{Int}} # Step at given point
#     subdomain_ranges::Vector{StepRange{Int,Int}} # Full range of indices per subdomain per element
# end

# function add!(slvh::SubLocalVariableHandler, info::InternalVariableInfo, qr::QuadratureRule, nel::Int)
#     @assert info.name ∉ slvh.names "Trying to register local variable $(info.name) twice. Registered variables: $(slvh.names)."

#     push(slvh.names, info.name)
#     nqp = length(qr.points)
#     local_range = if length(slvh.subdomain_ranges) > 0
#         o = last(last(slvh.subdomain_ranges))
#         (o+1):(o+1+info.size)
#     else
#         1:info.size
#     end
#     push!(slvh.local_ranges, local_range)

#     local_range = if length(slvh.subdomain_ranges) > 0
#         o = last(last(slvh.subdomain_ranges))
#         (o+1):(o+1+info.size)
#     else
#         1:info.size
#     end
# end

# """
#     LocalVariableHandler(...)

# Handler for variables without associated field. Also called "internal variable".
# """
# struct LocalVariableHandler{M} #<: AbstractDofHandler
#     mesh::M
#     sublvhandlers::Vector{SubLocalVariableHandler} # Must mirror corresponding SubDofHandler index structure
# end

# SubLocalVariableHandler() = SubLocalVariableHandler(Symbol[], Vector{Vector{Int}}(), Vector{StepRange{Int,Int}}(), Vector{Vector{Int}}())
# LocalVariableHandler(mesh) = LocalVariableHandler(mesh, SubLocalVariableHandler[])
# ndofs(lvh::LocalVariableHandler) = length(lvh.subdomain_ranges) > 0 ? last(last(lvh.subdomain_ranges)) : 0

# function local_dofrange(elementid::Int, sym::Symbol, qp::QuadraturePoint)
#     # Well...
# end

# This is the easiest solution for now
# TODO optimize.
struct LocalVariableHandler{DH} <: AbstractDofHandler
    dh::DH
end
LocalVariableHandler(mesh::SimpleMesh) = LocalVariableHandler(DofHandler(mesh))
Ferrite.close!(lvh::LocalVariableHandler) = close!(lvh.dh)
Ferrite.ndofs(lvh::LocalVariableHandler) = ndofs(lvh.dh)

# Utils to visualize local variables
struct QuadratureInterpolation{RefShape, QR <: QuadratureRule{RefShape}} <: Ferrite.ScalarInterpolation{RefShape, -1}
    qr::QR
end

Ferrite.getnbasefunctions(ip::QuadratureInterpolation) = getnquadpoints(ip.qr)
Ferrite.n_components(ip::QuadratureInterpolation)   = 1
Ferrite.n_dbc_components(::QuadratureInterpolation) = 0
Ferrite.adjust_dofs_during_distribution(::QuadratureInterpolation) = false
Ferrite.volumedof_interior_indices(ip::QuadratureInterpolation) = ntuple(i->i, getnbasefunctions(ip))
Ferrite.is_discontinuous(::Type{<:QuadratureInterpolation}) = true

function Ferrite.reference_coordinates(ip::QuadratureInterpolation)
    return [qp for i in 1:ip.num_components for qp in getpoints(ip.qr)]
end

function Ferrite.reference_shape_value(ip::QuadratureInterpolation, ::Vec, i::Int)
    throw(ArgumentError("shape function evaluation for interpolation $ip not implemented yet"))
end

function add_subdomain!(lvh::LocalVariableHandler, name::String, ivis::Vector{InternalVariableInfo}, qrc::QuadratureRuleCollection, compatible_dh::DofHandler)
    (; dh) = lvh
    mesh   = get_grid(dh)
    cells = mesh.grid.cells
    haskey(mesh.volumetric_subdomains, name) || error("Volumetric Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.volumetric_subdomains))")
    for (celltype, cellset) in mesh.volumetric_subdomains[name].data
        sdh = SubDofHandler(dh, _compatible_cellset(compatible_dh, first(cellset).idx))
        qr = getquadraturerule(qrc, sdh)
        for ivi in ivis
            add!(sdh, ivi.name, QuadratureInterpolation(qr)^ivi.size)
        end
    end
end
