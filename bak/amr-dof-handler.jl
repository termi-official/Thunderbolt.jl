struct AMRDofHandler{sdim,T,G<:Ferrite.AbstractGrid{sdim}} <: AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    field_interpolations::Vector{Interpolation}
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    grid::G
    ndofs::ScalarWrapper{Int}
end

function AMRDofHandler(grid::AbstractGrid)
    # isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. Use MixedDofHandler instead of DofHandler")
    AMRDofHandler(Symbol[], Int[], Interpolation[], Int[], Int[], grid, Ferrite.ScalarWrapper(0))
end
