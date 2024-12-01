using Ferrite, StaticArrays, ElasticArrays, Tensors

struct KoppCell{Dim, T <: Integer} <: Ferrite.AbstractCell{Ferrite.RefHypercube{Dim}}
    parent::T
    sequence::T
end

function Ferrite._distribute_dofs_for_cell!(dh::DofHandler, cell::KoppCell, ip_info::Ferrite.InterpolationInfo, nextdof::Int, vertexdict, edgedict, facedict) 
    cell_idx = cell.parent < 0 ? -cell.parent : cell.parent + (seq & (1 << (Dim + 1)) - 1 )
    return Ferrite._distribute_dofs_for_cell!(dh, dh.grid.base_grid.cells[cell_idx], ip_info, nextdof, vertexdict, edgedict, facedict) 
end


struct KoppTopology{T <: Integer, V} <: Ferrite.AbstractTopology
    root_idx::Vector{T}
    cell_facet_neighbors_offset::ElasticMatrix{Int, V}
    cell_facet_neighbors_length::ElasticMatrix{Int, V}
    neighbors::Vector{FacetIndex}
    root_facet_orientation_info::Vector{Ferrite.OrientationInfo} # indexed as root_idx + facet_idx - 1
end

@inline function _iterate(topology::KoppTopology, state::NTuple{2, Int} = (1,1)) where sdim
    state[1] > length(topology.cell_facet_neighbors_offset) && return nothing
    neighborhood_offset = topology.cell_facet_neighbors_offset[(state[1]-1) % 4 + 1, (state[1]-1) ÷ 4 + 1]
    neighborhood_length = topology.cell_facet_neighbors_length[(state[1]-1) % 4 + 1, (state[1]-1) ÷ 4 + 1]
    neighborhood_offset == 0 && return(FacetIndex(0,0), (state[1] + 1, state[2]))
    neighborhood = @view topology.neighbors[neighborhood_offset : neighborhood_offset + neighborhood_length - 1]
    if state[2] < neighborhood_length
        ret = neighborhood[state[2]]
        state = (state[1], state[2] + 1)
    else
        ret = neighborhood[state[2]]
        state = (state[1] + 1, 1)
    end
    return (ret, state)
end


function Base.iterate(topology::KoppTopology, state::NTuple{2, Int} = (1,1)) where sdim
    _iterate(topology, state)
end

function get_refinement_level(cell::KoppCell{Dim, T}) where {Dim, T}
    level::T = 0
    nbits::T = Dim + 1
    mask = (1 << nbits) - 1 # Does this need to be T too?
    maximum_level::T = sizeof(T)*8 ÷ nbits # Maybe use ÷ ?
    for level::T in 0:maximum_level # There should be an easier way
        cell.sequence & (mask << (nbits * level)) == 0 && return level 
    end
    return maximum_level
end

struct KoppRefinementCache{IntT <:Int}
    children_updated_indices::Vector{IntT}
    interfaces_updated_indices::Vector{IntT}
    marked_for_refinement::BitVector
    marked_for_coarsening::BitVector
    ncoarseninglevels::IntT
end

function KoppRefinementCache(topology::KoppTopology)
    return KoppRefinementCache(
        collect(1:length(topology.root_idx)),
        collect(1:length(topology.neighbors)÷2),
        falses(length(topology.root_idx)),
        falses(length(topology.root_idx)),
        1)
end

struct KoppValues{CellValuesT <: CellValues, FacetValuesT <: FacetValues, InterfaceValuesT <: InterfaceValues}
    cell_values::CellValuesT
    facet_values::FacetValuesT
    interface_values::InterfaceValuesT
end


struct KoppCache{V}
    cell_matrices::ElasticArray{Float64, 3, 2, V}
    interface_matrices::ElasticArray{Float64, 3, 2, V}
    u::Vector{Float64}
    interface_matrix_index::Vector{Int} 
    ansatz_isactive::BitVector
end

function _interface_matrices_offset_ctor(dh::Ferrite.AbstractDofHandler, refinement_cache::KoppRefinementCache, topology::KoppTopology{NFacets}, idx, offset) where NFacets
    neighbor
end

struct KoppGrid{Dim, G <: Ferrite.AbstractGrid{Dim}} <: Ferrite.AbstractGrid{Dim}
    base_grid::G
    kopp_cells::Vector{KoppCell{Dim, Int}}
end

@inline Ferrite.getncells(grid::KoppGrid) = length(grid.kopp_cells)
@inline Ferrite.getcells(grid::KoppGrid, v::Union{Int, Vector{Int}}) = grid.kopp_cells[v]
# @inline getcells(grid::KoppGrid, setname::String) = grid.cells[collect(getcellset(grid,setname))]
@inline Ferrite.getcelltype(grid::KoppGrid) = eltype(grid.kopp_cells)
@inline Ferrite.getcelltype(grid::KoppGrid, i::Int) = typeof(grid.kopp_cells[i])

@inline Ferrite.getrefshape(::KoppCell{Dim}) where Dim = Ferrite.RefHypercube{Dim}

@inline Ferrite.getnnodes(grid::KoppGrid) = length(grid.base_grid.nodes)

function Ferrite.nnodes_per_cell(grid::KoppGrid)
    if !isconcretetype(getcelltype(grid))
        error("There are different celltypes in the `grid`. Use `nnodes_per_cell(grid, cellid::Int)` instead")
    end
    return Ferrite.nnodes(first(grid.base_grid.cells))
end
# TODO: URGENT FIX ME... HOW?
@inline Ferrite.nnodes_per_cell(grid::KoppGrid, i::Int) = Ferrite.nnodes(grid.base_grid.cells[1])

function Ferrite.CellCache(grid::KoppGrid{dim}, flags::UpdateFlags=Ferrite.UpdateFlags(false, true, false)) where {dim}
    N = Ferrite.nnodes_per_cell(grid, 1) # nodes and coords will be resized in `reinit!`
    nodes = Int[]
    coords = zeros(Vec{dim, Float64}, N)
    return Ferrite.CellCache(flags, grid, -1, nodes, coords, nothing, Int[])
end

function Ferrite.CellCache(dh::DofHandler{dim, <: KoppGrid}, flags::UpdateFlags=Ferrite.UpdateFlags(false, true, true)) where {dim}
    n = Ferrite.ndofs_per_cell(dh.subdofhandlers[1]) # dofs and coords will be resized in `reinit!`
    N = Ferrite.nnodes_per_cell(Ferrite.get_grid(dh), 1)
    nodes = Int[]
    coords = zeros(Vec{dim, Float64}, N)
    celldofs = zeros(Int, n)
    return Ferrite.CellCache(flags, Ferrite.get_grid(dh), -1, nodes, coords, dh, celldofs)
end

function Ferrite.CellCache(sdh::SubDofHandler{<:DofHandler{dim, <: KoppGrid}}, flags::Ferrite.UpdateFlags=UpdateFlags()) where dim
    Tv = Float64
    Ferrite.CellCache(flags, sdh.dh.grid, -1, Int[], Tv[], sdh, Int[])
end


function Ferrite.CellIterator(gridordh::Union{KoppGrid, <:DofHandler{Dim, <:KoppGrid}},
    set::Union{Ferrite.IntegerCollection,Nothing}=nothing,
    flags::UpdateFlags=UpdateFlags(false, true, true)) where Dim
    if set === nothing
        grid = gridordh isa DofHandler ? Ferrite.get_grid(gridordh) : gridordh
        set = 1:getncells(grid)
    end
    if gridordh isa DofHandler
        # TODO: Since the CellCache is resizeable this is not really necessary to check
        #       here, but might be useful to catch slow code paths?
        # _check_same_celltype(get_grid(gridordh), set)
    end
    return Ferrite.CellIterator(CellCache(gridordh, flags), set)
end
function Ferrite.CellIterator(gridordh::Union{KoppGrid, <:DofHandler{Dim, <:KoppGrid}}, flags::UpdateFlags) where Dim
    return Ferrite.CellIterator(gridordh, nothing, flags)
end

function Ferrite.get_grid(dh::Ferrite.AbstractDofHandler)
    return dh.grid
end

function Ferrite.InterfaceIterator(gridordh::Union{Ferrite.AbstractGrid, Ferrite.AbstractDofHandler},
        topology::Ferrite.AbstractTopology = KoppTopology(gridordh isa Ferrite.AbstractGrid ? gridordh : Ferrite.get_grid(gridordh)))
    grid = gridordh isa Ferrite.AbstractGrid ? gridordh : Ferrite.get_grid(gridordh)
    return Ferrite.InterfaceIterator(Ferrite.InterfaceCache(gridordh), grid, topology)
end

function Base.iterate(ii::Ferrite.InterfaceIterator{IC, <:KoppGrid{sdim}}, state::NTuple{3, Int} = (1,1,1)) where {sdim, IC}
    ninterfaces = state[3]
    facet_idx = state[1]
    nneighbors = state[2]
    while true
        facet_idx > length(ii.topology.cell_facet_neighbors_offset) && return nothing
        neighborhood_offset = ii.topology.cell_facet_neighbors_offset[(facet_idx-1) % (2*sdim) + 1, (facet_idx-1) ÷ (2*sdim) + 1]
        neighborhood_length = ii.topology.cell_facet_neighbors_length[(facet_idx-1) % (2*sdim) + 1, (facet_idx-1) ÷ (2*sdim) + 1]
        if neighborhood_offset == 0
           facet_idx = facet_idx + 1
           continue
        end
        neighborhood = @view ii.topology.neighbors[neighborhood_offset : neighborhood_offset + neighborhood_length - 1]
        if nneighbors < neighborhood_length
            neighbor = neighborhood[nneighbors]
            facet_idx = facet_idx
            nneighbors = nneighbors + 1
        else
            neighbor = neighborhood[nneighbors]
            facet_idx = facet_idx + 1
            nneighbors = 1
        end
        _facet_idx = nneighbors == 1 ? facet_idx - 1 : facet_idx
        _nneighbors = nneighbors == 1 ? nneighbors : nneighbors - 1
        cell_idx = (_facet_idx - 1) ÷ (2*sdim) + 1
        facet_a = (_facet_idx - 1) % (2*sdim) + 1
        cell = ii.grid.kopp_cells[cell_idx]
        cell_refinement_level = get_refinement_level(cell)
        neighbor_refinement_level = get_refinement_level(ii.grid.kopp_cells[neighbor[1]])
        if ((cell_refinement_level == neighbor_refinement_level) && (neighbor[1] > cell_idx)) || (cell_refinement_level < neighbor_refinement_level)
            reinit!(ii.cache, FacetIndex(cell_idx, facet_a), neighbor)
            ninterfaces += 1
            return (facet_idx, nneighbors, ninterfaces), (facet_idx, nneighbors, ninterfaces)
        else
            continue
        end
    end
    return nothing
end


function Ferrite.generate_grid(C::Type{KoppCell{2, T}}, nel::NTuple{2,Int}) where {T}
    grid =  generate_grid(Quadrilateral, nel)
    return KoppGrid(grid)
end

function Ferrite.generate_grid(C::Type{KoppCell{3, T}}, nel::NTuple{3,Int}) where {T}
    grid =  generate_grid(Hexahedron, nel)
    return KoppGrid(grid)
end

function KoppTopology(grid::KoppGrid{Dim}) where Dim
    base_topology = ExclusiveTopology(grid.base_grid)
    root_idx = collect(1:getncells(grid.base_grid))
    neighbors = collect(reinterpret(FacetIndex, reduce(vcat, Ferrite._get_facet_facet_neighborhood(base_topology, Val(Dim)))))
    cell_facet_neighbors_offset = ElasticMatrix(transpose([isempty(neighbor) ? 0 : findfirst(==(FacetIndex(neighbor[1][1], neighbor[1][2])), neighbors) for neighbor in Ferrite._get_facet_facet_neighborhood(base_topology, Val(Dim))]))
    cell_facet_neighbors_length = ElasticMatrix(transpose([isempty(neighbor) ? 0 : 1 for neighbor in Ferrite._get_facet_facet_neighborhood(base_topology, Val(Dim))]))
    root_facet_orientation_info = reduce(vcat, [Ferrite.OrientationInfo(Ferrite.facets(cell)[facet]) for facet in 1:2Dim, cell in getcells(grid.base_grid)])
    return KoppTopology(root_idx, cell_facet_neighbors_offset, cell_facet_neighbors_length, neighbors, root_facet_orientation_info)
end

function Ferrite.getneighborhood(topology::KoppTopology, grid::KoppGrid, facetidx::FacetIndex)
    offset = topology.cell_facet_neighbors_offset[facetidx[2], facetidx[1]]
    length = topology.cell_facet_neighbors_length[facetidx[2], facetidx[1]]
    return offset == 0 ? nothing : @view topology.neighbors[offset : offset + length - 1]
end

function KoppGrid(grid::G) where {Dim, G <: Ferrite.AbstractGrid{Dim}}
    cells = KoppCell{Dim, Int}[KoppCell{Dim, Int}(-i, 0) for i in 1:getncells(grid)]
    return KoppGrid(grid, cells)
end
function KoppCache(grid::KoppGrid, dh::Ferrite.AbstractDofHandler, kopp_values::KoppValues, refinement_cache::KoppRefinementCache, topology::KoppTopology{NFacets}, u = zeros(ndofs(dh))) where NFacets
    @assert Ferrite.isclosed(dh) "DofHandler must be closed" 
    ndofs_cell = ndofs_per_cell(dh)
    ref_shape = getrefshape(dh.grid.kopp_cells[1])
    ip = DiscontinuousLagrange{ref_shape, 1}()
    qr = QuadratureRule{ref_shape}(1)
    geo_mapping = Ferrite.GeometryMapping{1}(Float64, ip, qr)
    cell_matrices = ElasticArray(zeros(Float64, ndofs_cell, ndofs_cell, length(refinement_cache.children_updated_indices))) 
    for cell_cache in Ferrite.CellIterator(grid)
        cell_idx = cell_cache.cellid
        assemble_element_matrix!(cell_matrices[:,:,cell_idx], kopp_values)
    end
    interface_matrices = ElasticArray(zeros(Float64, 2 * ndofs_cell, 2 * ndofs_cell, length(refinement_cache.interfaces_updated_indices))) 
    # Construct interface matrix indices
    interface_matrix_indices = Vector{Int}(undef, count(topology.cell_facet_neighbors_length .!= 0 ))
    new_offset = 1
    for (idx, offset) in pairs(IndexCartesian(), topology.cell_facet_neighbors_offset) 
        offset == 0 && continue
        neighbor_cell = topology.neighbors[offset][1]
        neighbor_facet = topology.neighbors[offset][2]
        current_cell = idx[2]
        if(neighbor_cell < current_cell)
            interface_matrix_indices[offset] = interface_matrix_indices[topology.cell_facet_neighbors_offset[neighbor_facet, neighbor_cell]]
            continue
        end
        interface_matrix_indices[offset] = new_offset
        new_offset += 1
    end
    ansatz_isactive = trues(ndofs(dh))
    return KoppCache(cell_matrices, interface_matrices, u, interface_matrix_indices, ansatz_isactive)
end

include("utils.jl")

function refine!(
        grid::KoppGrid{Dim},
        topology::KoppTopology,
        refinement_cache::KoppRefinementCache,
        kopp_cache::KoppCache,
        kopp_values::KoppValues,
        cellset::Set{CellIndex},
        dh::DofHandler) where Dim
    n_refined_cells = length(cellset)
    new_length = length(grid.kopp_cells)+(2^Dim)*n_refined_cells
    # we need to resize this one early
    _resize_marked_for_refinement!(grid, refinement_cache, cellset)
    n_refined_interfaces = 0
    NFacets = 2*Dim
    refinement_cache.children_updated_indices .= 1:length(refinement_cache.children_updated_indices)
    # Counting how many refined cells and interfaces and calculating the new index
    _update_refinement_cache_isactive!(grid, topology, refinement_cache, kopp_cache, cellset, dh)

    temp_topology = copy_topology(topology)
    temp_grid = deepcopy(grid)
    _resize_topology!(topology, new_length, Val(Dim))
    
    zero_topology!(topology)

    n_neighborhoods = count_neighbors_update_indexing!(grid, topology, temp_topology, refinement_cache)

    # Resizing the vectors
    temp_celldofs = copy(dh.cell_dofs)
    temp_cell_dofs_offset = copy(dh.cell_dofs_offset)
    temp_cell_to_subdofhandler = copy(dh.cell_to_subdofhandler)
    
    ndofs_old = dh.ndofs
    
    _resize_bunch_of_stuff!(grid, topology, kopp_cache, dh, n_neighborhoods, new_length)

    update_cells!(grid, refinement_cache)

    update_root_idx!(grid, topology, temp_topology, refinement_cache)

    update_dofs!(grid, refinement_cache, dh, temp_celldofs, temp_cell_dofs_offset, temp_cell_to_subdofhandler)

    kopp_cache.ansatz_isactive[ndofs_old : end] .= true
    kopp_cache.interface_matrix_index .= 0


    # # deactivate all parents' shape functions
    update_neighbors!(grid, temp_grid, topology, temp_topology, refinement_cache)

    # # Refine KoppCache
    resize!(kopp_cache.u, new_length * dh.subdofhandlers[1].ndofs_per_cell)

    update_koppcache!(grid, refinement_cache, topology, temp_topology, kopp_cache, kopp_values, dh, NFacets)

    # # # resize refinement cache
    resize!(refinement_cache.children_updated_indices, new_length)
    resize!(refinement_cache.marked_for_refinement, new_length)
    resize!(refinement_cache.marked_for_coarsening, new_length)
    # resize!(refinement_cache.interfaces_updated_indices, length(kopp_cache.interface_matrices) + n_refined_interfaces * (2^(Dim-1) - 1))
    refinement_cache.marked_for_refinement .= false
    refinement_cache.marked_for_coarsening .= false
    return nothing
end

function coarsen!(
        grid::KoppGrid{Dim},
        topology::KoppTopology,
        refinement_cache::KoppRefinementCache,
        kopp_cache::KoppCache,
        kopp_values::KoppValues,
        cellset::Set{CellIndex},
        dh::DofHandler) where Dim
    n_coarsened_cells = length(cellset)
    new_length = length(grid.kopp_cells)-(2^Dim)*n_coarsened_cells
    # we need to resize this one early
    __resize_marked_for_refinement!(grid, refinement_cache, cellset)
    # n_refined_interfaces = 0
    NFacets = 2*Dim
    refinement_cache.children_updated_indices .= 1:length(refinement_cache.children_updated_indices)
    # Counting how many refined cells and interfaces and calculating the new index
    __update_refinement_cache_isactive!(grid, topology, refinement_cache, kopp_cache, cellset, dh)

    temp_topology = copy_topology(topology)
    temp_grid = deepcopy(grid)
    _resize_topology!(topology, new_length, Val(Dim))
    
    zero_topology!(topology)

    n_neighborhoods = do_the_sets_thing(grid, topology, temp_topology, refinement_cache)

    # # Resizing the vectors
    temp_celldofs = copy(dh.cell_dofs)
    temp_cell_dofs_offset = copy(dh.cell_dofs_offset)
    temp_cell_to_subdofhandler = copy(dh.cell_to_subdofhandler)
    
    ndofs_old = dh.ndofs
    
    resize!(topology.neighbors, n_neighborhoods)
   resize!(topology.root_idx, new_length)

    update_coarsened_cells!(grid, refinement_cache)
    resize!(grid.kopp_cells, new_length)

    update_coarsened_root_idx!(grid, topology, temp_topology, refinement_cache)

    update_coarsened_dofs!(grid, refinement_cache, dh, temp_celldofs, temp_cell_dofs_offset, temp_cell_to_subdofhandler)
    resize!(dh.cell_dofs, new_length * dh.subdofhandlers[1].ndofs_per_cell)
    resize!(dh.cell_dofs_offset, new_length)
    resize!(dh.cell_to_subdofhandler, new_length)
    kopp_cache.interface_matrix_index .= 0

    # # Refine KoppCache
    resize!(kopp_cache.u, new_length * dh.subdofhandlers[1].ndofs_per_cell)

    update_coarsened_koppcache!(grid, refinement_cache, topology, temp_topology, kopp_cache, kopp_values, dh, NFacets, Dim)
    resize!(kopp_cache.interface_matrix_index, n_neighborhoods)
    resize!(kopp_cache.interface_matrices,size(kopp_cache.interface_matrices)[1], size(kopp_cache.interface_matrices)[2], n_neighborhoods ÷ 2)
    resize!(kopp_cache.cell_matrices, size(kopp_cache.cell_matrices)[1], size(kopp_cache.cell_matrices)[2], new_length)
    resize!(kopp_cache.ansatz_isactive, new_length * dh.subdofhandlers[1].ndofs_per_cell)

    # # # # resize refinement cache
    resize!(refinement_cache.children_updated_indices, new_length)
    resize!(refinement_cache.marked_for_coarsening, new_length)
    resize!(refinement_cache.marked_for_refinement, new_length)
    # resize!(refinement_cache.interfaces_updated_indices, length(kopp_cache.interface_matrices) + n_refined_interfaces * (2^(Dim-1) - 1))
    refinement_cache.marked_for_coarsening .= false
    return nothing
end
function Ferrite.getcoordinates!(coords, grid::KoppGrid, i)
    coords .= reinterpret(Vec{2, Float64},SVector(get_refined_coords(grid, i)))
    return nothing
end

# # # 2 allocs for J?
function assemble_element_matrix!(Ke, kopp_values::KoppValues)
    cv = kopp_values.cell_values
    n_basefuncs = getnbasefunctions(cv)
    for q_point in 1:getnquadpoints(cv.qr)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δu  = Ferrite.shape_value(cv, q_point, i)
            for j in 1:n_basefuncs
                u = Ferrite.shape_value(cv, q_point, i)
                Ke[i, j] += (δu ⋅ u) * dΩ
            end
        end
    end
    return nothing 
end
function Ferrite.reinit!(iv::InterfaceValues, ic::InterfaceCache{<:FacetCache{<:CellCache{<:Any, <:KoppGrid{Dim}}}}, topology::KoppTopology) where Dim
    root_here = topology.root_idx[cellid(ic.a)]
    root_there = topology.root_idx[cellid(ic.b)]
    facet_here = ic.a.current_facet_id[]
    facet_there = ic.b.current_facet_id[]

    OI_here = topology.root_facet_orientation_info[(root_here-1) *2Dim  + facet_here ]
    OI_there = topology.root_facet_orientation_info[(root_there-1) *2Dim + facet_there ]
    flipped = root_here == root_there ? true : OI_here.flipped ⊻ OI_there.flipped
    shift_index = OI_there.shift_index - OI_here.shift_index
    interface_transformation = Ferrite.InterfaceOrientationInfo{Ferrite.RefHypercube{Dim}, Ferrite.RefHypercube{Dim}}(flipped, shift_index, OI_there.shift_index, facet_here, facet_there)
    return Ferrite.reinit!(iv,
        getcells(ic.a.cc.grid, cellid(ic.a)),
        getcoordinates(ic.a),
        ic.a.current_facet_id[],
        getcells(ic.b.cc.grid, cellid(ic.b)),
        getcoordinates(ic.b),
        ic.b.current_facet_id[],
        interface_transformation
        )
end
function Ferrite.reinit!(
    iv::InterfaceValues,
    cell_here::KoppCell, coords_here::AbstractVector{Vec{dim, T}}, facet_here::Int,
    cell_there::KoppCell, coords_there::AbstractVector{Vec{dim, T}}, facet_there::Int,
    interface_transformation
) where {dim, T}

    # reinit! the here side as normal
    Ferrite.reinit!(iv.here, cell_here, coords_here, facet_here)
    dim == 1 && return reinit!(iv.there, cell_there, coords_there, facet_there)
    # Transform the quadrature points from the here side to the there side
    Ferrite.set_current_facet!(iv.there, facet_there) # Includes boundscheck

    quad_points_a = Ferrite.getpoints(iv.here.fqr, facet_here)
    quad_points_b = Ferrite.getpoints(iv.there.fqr, facet_there)
    Ferrite.transform_interface_points!(quad_points_b, quad_points_a, interface_transformation)
    seq_length = get_refinement_level(cell_there) - get_refinement_level(cell_here)
    _transform_to_parent!(quad_points_b, (Ferrite.reference_coordinates(Lagrange{Ferrite.RefHypercube{dim},1}())), cell_there.sequence & ((1 << ((dim + 1) * seq_length)) - 1 ))
    # TODO: This is the bottleneck, cache it?
    @assert length(quad_points_a) <= length(quad_points_b)

    # Re-evaluate shape functions in the transformed quadrature points
    Ferrite.precompute_values!(Ferrite.get_fun_values(iv.there),  quad_points_b)
    Ferrite.precompute_values!(Ferrite.get_geo_mapping(iv.there), quad_points_b)

    # reinit! the "there" side
    Ferrite.reinit!(iv.there, cell_there, coords_there, facet_there)
    return iv
end

function assemble_interface_matrix!(Ki, kopp_values::KoppValues, μ::Float64 = 10.)
    iv = kopp_values.interface_values
    for q_point in 1:getnquadpoints(iv)
        # Get the normal to facet A
        normal = getnormal(iv, q_point)
        # Get the quadrature weight
        dΓ = getdetJdV(iv, q_point)
        # Loop over test shape functions
        for i in 1:getnbasefunctions(iv)
            # Multiply the jump by the negative normal to get the definition from the theory section.
            δu_jump = shape_value_jump(iv, q_point, i) * (-normal)
            ∇δu_avg = shape_gradient_average(iv, q_point, i)
            # Loop over trial shape functions
            for j in 1:getnbasefunctions(iv)
                # Multiply the jump by the negative normal to get the definition from the theory section.
                u_jump = shape_value_jump(iv, q_point, j) * (-normal)
                ∇u_avg = shape_gradient_average(iv, q_point, j)
                # Add contribution to Ki
                Ki[i, j] += -(δu_jump ⋅ ∇u_avg + ∇δu_avg ⋅ u_jump)*dΓ +  μ * (δu_jump ⋅ u_jump) * dΓ
            end
        end
    end
end