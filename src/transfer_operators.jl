abstract type AbstractTransferOperator end

# TODO remove after https://github.com/Ferrite-FEM/Ferrite.jl/pull/820 is merged
"""
    _spatial_coordinate(ip::VectorizedInterpolation, ξ::Vec, cell_coordinates::AbstractVector{<:Vec{sdim, T}})
Compute the spatial coordinate in a given quadrature point. `cell_coordinates` contains the nodal coordinates of the cell.
The coordinate is computed, using the geometric interpolation, as
``\\mathbf{x} = \\sum\\limits_{i = 1}^n M_i (\\mathbf{\\xi}) \\mathbf{\\hat{x}}_i``
"""
_spatial_coordinate(ip::VectorizedInterpolation, ξ::Vec, cell_coordinates::AbstractVector{<:Vec{sdim, T}}) where {T, sdim} = spatial_coordinate(ip.ip, ξ, cell_coordinates)

function _spatial_coordinate(interpolation::ScalarInterpolation, ξ::Vec{<:Any,T}, cell_coordinates::AbstractVector{<:Vec{sdim, T}}) where {T, sdim}
    n_basefuncs = getnbasefunctions(interpolation)
    @boundscheck checkbounds(cell_coordinates, Base.OneTo(n_basefuncs))

    x = zero(Vec{sdim, T})
    @inbounds for j in 1:n_basefuncs
        M = shape_value(interpolation, ξ, j)
        x += M * cell_coordinates[j]
    end
    return x
end


"""
    NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name::Symbol)

Construct a transfer operator to move a field `field_name` from dof handler `dh_from` to another
dof handler `dh_to`, assuming that all spatial coordinates of the dofs for `dh_to` are in the
interior or boundary of the mesh contained within dh_from. This is necessary to have valid
interpolation values, as this operator does not have extrapolation functionality.

!!! note
    We assume a continuous coordinate field, if the interpolation of the named field is continuous.
"""
struct NodalIntergridInterpolation{PH <: PointEvalHandler, DH1 <: AbstractDofHandler, DH2 <: AbstractDofHandler} <: AbstractTransferOperator
    ph::PH
    dh_from::DH1
    dh_to::DH2
    node_to_dof_map::Vector{Int}
    dof_to_node_map::Dict{Int,Int}
    field_name::Symbol

    function NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name::Symbol) where sdim
        @assert field_name ∈ dh_to.field_names
        @assert field_name ∈ dh_from.field_names

        dofset = Set{Int}()
        for sdh in dh_to.subdofhandlers
            # Skip subdofhandler if field is not present
            field_name ∈ sdh.field_names || continue
            # Just gather the dofs of the given field in the set
            for cellidx ∈ sdh.cellset
                dofs = celldofs(dh_to, cellidx)
                for dof in dofs[dof_range(sdh, field_name)]
                    push!(dofset, dof)
                end
            end
        end
        node_to_dof_map = sort(collect(dofset))

        # Build inverse map
        dof_to_node_map = Dict{Int,Int}()
        next_dof_index = 1
        for dof ∈ node_to_dof_map
            dof_to_node_map[dof] = next_dof_index
            next_dof_index += 1
        end

        # Compute nodes
        nodes = Vector{Ferrite.get_coordinate_type(dh_to.grid)}(undef, length(dofset))
        for sdh in dh_to.subdofhandlers
            # Skip subdofhandler if field is not present
            field_name ∈ sdh.field_names || continue
            # Grab the reference coordinates of the field to interpolate
            ip = Ferrite.getfieldinterpolation(sdh, field_name)
            ref_coords = Ferrite.reference_coordinates(ip)
            # Grab the geometric interpolation
            gip = Ferrite.default_interpolation(typeof(getcells(Ferrite.get_grid(dh_to), first(sdh.cellset))))
            for cc ∈ CellIterator(sdh)
                # Compute for each dof the spatial coordinate of from the reference coordiante and store.
                # NOTE We assume a continuous coordinate field if the interpolation is continuous.
                dofs = celldofs(cc)[dof_range(sdh, field_name)]
                for (dofidx,dof) in enumerate(dofs)
                    nodes[dof_to_node_map[dof]] = _spatial_coordinate(gip, ref_coords[dofidx], getcoordinates(cc))
                end
            end
        end

        ph = PointEvalHandler(Ferrite.get_grid(dh_from), nodes)

        n_missing = sum(x -> x === nothing, ph.cells)
        n_missing == 0 || @warn "Constructing the interpolation for $field_name failed. $n_missing points not found."

        new{typeof(ph), typeof(dh_from), typeof(dh_to)}(
            ph,
            dh_from,
            dh_to,
            node_to_dof_map,
            dof_to_node_map,
            field_name
        )
    end
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(u_to::AbstractArray, operator::NodalIntergridInterpolation, u_from::AbstractArray)
    # TODO non-allocating version
    u_to[operator.node_to_dof_map] .= Ferrite.evaluate_at_points(operator.ph, operator.dh_from, u_from, operator.field_name)
end


#### Parameter <-> Solution
# TODO generalize these below into something like
# struct ParameterSyncronizer <: AbstractTransferOperator
#     parameter_buffer
#     index_range_global
#     method
# end
# struct IdentityTransferOpeator <: AbstractTransferOperator end
# syncronize_parameters!(integ, f, syncer::SimpleParameterSyncronizer) = transfer!(parameter_buffer.data, syncer.method, @view f.uparent[index_range_global])
# transfer!(y, syncer::IdentityTransferOpeator, x) = y .= x
"""
    Utility function to synchronize the volume in a split [`RSAFDQ2022Function`](@ref)
"""
struct VolumeTransfer0D3D{TP} <: AbstractTransferOperator
    tying::TP
end

function syncronize_parameters!(integ, f, syncer::VolumeTransfer0D3D)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for chamber ∈ syncer.tying.chambers
        chamber.V⁰ᴰval = integ.uparent[chamber.V⁰ᴰidx_global]
    end
end

"""
    Utility function to synchronize the pressire in a split [`RSAFDQ2022Function`](@ref)
"""
struct PressureTransfer3D0D{TP } <: AbstractTransferOperator
    tying::TP
end

function syncronize_parameters!(integ, f, syncer::PressureTransfer3D0D)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for (chamber_idx,chamber) ∈ enumerate(syncer.tying.chambers)
        p = integ.uparent[chamber.pressure_dof_index_global]
        # The pressure buffer is constructed in a way that the chamber index and
        # pressure index coincides
        f.p[chamber_idx] = p
    end
end