function _resize_marked_for_refinement!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache,
    cellset::Set{CellIndex}) where {Dim}
    n_refined_cells = length(cellset)
    new_length = length(grid.kopp_cells) + (2^Dim) * n_refined_cells
    resize!(refinement_cache.marked_for_refinement, new_length)
    refinement_cache.marked_for_refinement[length(grid.kopp_cells) + 1 : end] .= false
    return nothing
end

function __resize_marked_for_refinement!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache,
    cellset::Set{CellIndex}) where {Dim}
    n_refined_cells = length(cellset)
    new_length = length(grid.kopp_cells) - (2^Dim) * n_refined_cells
    resize!(refinement_cache.marked_for_refinement, new_length)
    refinement_cache.marked_for_refinement[length(grid.kopp_cells) + 1 : end] .= false
    return nothing
end

function _update_refinement_cache_isactive!(
    grid::KoppGrid{Dim},
    topology::KoppTopology, # Old topology
    refinement_cache::KoppRefinementCache,
    kopp_cache::KoppCache,
    cellset::Set{CellIndex},
    dh::DofHandler) where {Dim}
    for (i, cell) in enumerate(grid.kopp_cells)
        if CellIndex(i) ∈ cellset
            refinement_cache.children_updated_indices[i+1:end] .= (@view refinement_cache.children_updated_indices[i+1:end]) .+ 2^Dim
            refinement_cache.marked_for_refinement[refinement_cache.children_updated_indices[i]] = true
            kopp_cache.ansatz_isactive[@view dh.cell_dofs[dh.cell_dofs_offset[i]:dh.cell_dofs_offset[i]+dh.subdofhandlers[1].ndofs_per_cell-1]] .= false
            # kopp_cache.ansatz_isactive[dh.cell_dofs_offset[i]+dh.subdofhandlers[1].ndofs_per_cell : dh.cell_dofs_offset[i]+dh.subdofhandlers[1].ndofs_per_cell*(2^Dim) - 1] .= true
            for facet in 1:2Dim
                interface_offset = topology.cell_facet_neighbors_offset[facet, i]
                interface_length = topology.cell_facet_neighbors_length[facet, i]
                for interface_index_idx in interface_offset : interface_offset + interface_length - 1
                    interface_offset == 0 && continue
                    neighbor = topology.neighbors[interface_index_idx]
                    # Refine interface when the current cell is lower index than the neighbor only to avoid race conditions
                    get_refinement_level(grid.kopp_cells[neighbor.idx[1]]) == get_refinement_level(cell) && i > neighbor.idx[1] && continue
                    # Refine the interface only if the neighbor is not finer than the current cell
                    get_refinement_level(grid.kopp_cells[neighbor.idx[1]]) < get_refinement_level(cell) && continue
                    # TODO: Invalidate interface matrices
                    # grid.interfaces_recompute[interface:interface + 1] .= true
                    refinement_cache.interfaces_updated_indices[interface_index_idx+1:end] .= (@view refinement_cache.interfaces_updated_indices[interface_index_idx+1:end]) .+ (2^(Dim-1) - 1)
                    # n_refined_interfaces += 1
                end
            end
        end
    end
    return nothing
end

function __update_refinement_cache_isactive!(
    grid::KoppGrid{Dim},
    topology::KoppTopology, # Old topology
    refinement_cache::KoppRefinementCache,
    kopp_cache::KoppCache,
    cellset::Set{CellIndex},
    dh::DofHandler) where {Dim}
    for (i, cell) in enumerate(grid.kopp_cells)
        if CellIndex(i) ∈ cellset
            refinement_cache.children_updated_indices[i+(2^Dim)+1:end] .= (@view refinement_cache.children_updated_indices[i+(2^Dim)+1:end]) .- 2^Dim
            refinement_cache.children_updated_indices[i+1:i+(2^Dim)] .= 0
            refinement_cache.marked_for_coarsening[refinement_cache.children_updated_indices[i]] = true
            kopp_cache.ansatz_isactive[@view dh.cell_dofs[dh.cell_dofs_offset[i]:dh.cell_dofs_offset[i]+dh.subdofhandlers[1].ndofs_per_cell-1]] .= true
            # kopp_cache.ansatz_isactive[dh.cell_dofs_offset[i]+dh.subdofhandlers[1].ndofs_per_cell : dh.cell_dofs_offset[i]+dh.subdofhandlers[1].ndofs_per_cell*(2^Dim) - 1] .= true
            for facet in 1:2Dim
                interface_offset = topology.cell_facet_neighbors_offset[facet, i]
                interface_length = topology.cell_facet_neighbors_length[facet, i]
                for interface_index_idx in interface_offset : interface_offset + interface_length - 1
                    interface_offset == 0 && continue
                    neighbor = topology.neighbors[interface_index_idx]
                    # Refine interface when the current cell is lower index than the neighbor only to avoid race conditions
                    get_refinement_level(grid.kopp_cells[neighbor.idx[1]]) == get_refinement_level(cell) && i > neighbor.idx[1] && continue
                    # Refine the interface only if the neighbor is not finer than the current cell
                    get_refinement_level(grid.kopp_cells[neighbor.idx[1]]) < get_refinement_level(cell) && continue
                    # TODO: Invalidate interface matrices
                    # grid.interfaces_recompute[interface:interface + 1] .= true
                    refinement_cache.interfaces_updated_indices[interface_index_idx+1:end] .= (@view refinement_cache.interfaces_updated_indices[interface_index_idx+1:end]) .- (2^(Dim-1) - 1)
                    # n_refined_interfaces += 1
                end
            end
        end
    end
    return nothing
end

function copy_topology(topology::KoppTopology)
    temp_topology = deepcopy(topology)
    return temp_topology
end

function _resize_topology!(topology::KoppTopology, new_length::Int, ::Val{Dim}) where {Dim}
    resize!(topology.cell_facet_neighbors_offset, 2Dim, new_length)
    resize!(topology.cell_facet_neighbors_length, 2Dim, new_length)
    return nothing
end

function zero_topology!(topology::KoppTopology)
    topology.cell_facet_neighbors_offset .= 0
    topology.cell_facet_neighbors_length .= 0
    return nothing
end

function do_the_sets_thing(
    grid::KoppGrid{Dim},
    topology::KoppTopology,
    temp_topology::KoppTopology,
    refinement_cache::KoppRefinementCache) where {Dim}
    NFacets = 2 * Dim
    temp_cell_facet_neighbors_offset = temp_topology.cell_facet_neighbors_offset
    temp_cell_facet_neighbors_length = temp_topology.cell_facet_neighbors_length
    n_neighborhoods = 0
    ref_cell_neighborhood = get_ref_cell_neighborhood(getrefshape(getcelltype(grid.base_grid)))
    set = Set{FacetIndex}()
    sizehint!(set, maximum(temp_cell_facet_neighbors_length)*(2^(Dim-1)))
    empty!(topology.neighbors)
    for (i, cell) in enumerate(grid.kopp_cells)
        new_i = refinement_cache.children_updated_indices[i]
        new_i == 0 && continue
        if refinement_cache.marked_for_coarsening[new_i]
            # TODO: Avoid using sets
            for facet in 1:NFacets
                for new_seq in 1:2^Dim
                    if ref_cell_neighborhood[new_seq][facet] == 0
                        child_neighbors_offset = temp_cell_facet_neighbors_offset[facet, i+new_seq]
                        child_neighbors_length = temp_cell_facet_neighbors_length[facet, i+new_seq]
                        child_neighbors = @view temp_topology.neighbors[child_neighbors_offset : child_neighbors_offset + child_neighbors_length - 1]
                        for child_neighbor in child_neighbors
                            parent_idx = grid.kopp_cells[child_neighbor[1]].parent
                            if parent_idx < 0 || !refinement_cache.marked_for_coarsening[parent_idx]
                                push!(set, FacetIndex(refinement_cache.children_updated_indices[child_neighbor[1]], child_neighbor[2]))
                            else
                                push!(set, FacetIndex(parent_idx, child_neighbor[2]))
                            end
                        end
                    end
                end
                topology.cell_facet_neighbors_offset[facet, new_i] = length(set) == 0 ? 0 : n_neighborhoods + 1
                topology.cell_facet_neighbors_length[facet, new_i] = length(set) == 0 ? 0 : length(set)
                n_neighborhoods += length(set)
                append!(topology.neighbors, set)
                empty!(set)
            end
        else
            for facet in 1:NFacets
                child_neighbors_offset = temp_cell_facet_neighbors_offset[facet, i]
                child_neighbors_length = temp_cell_facet_neighbors_length[facet, i]
                child_neighbors = @view temp_topology.neighbors[child_neighbors_offset : child_neighbors_offset + child_neighbors_length - 1]
                for child_neighbor in child_neighbors
                    parent_idx = grid.kopp_cells[child_neighbor[1]].parent
                    if parent_idx < 0 || !refinement_cache.marked_for_coarsening[parent_idx]
                        push!(set, FacetIndex(refinement_cache.children_updated_indices[child_neighbor[1]], child_neighbor[2]))
                    else
                        push!(set, FacetIndex(parent_idx, child_neighbor[2]))
                    end
                end
                topology.cell_facet_neighbors_offset[facet, new_i] = length(set) == 0 ? 0 : n_neighborhoods + 1
                topology.cell_facet_neighbors_length[facet, new_i] = length(set) == 0 ? 0 : length(set)
                n_neighborhoods += length(set)
                append!(topology.neighbors, set)
                empty!(set)
            end
        end
    end
    return n_neighborhoods
end

function count_neighbors_update_indexing!(
    grid::KoppGrid{Dim},
    topology::KoppTopology,
    temp_topology::KoppTopology,
    refinement_cache::KoppRefinementCache) where {Dim}
    NFacets = 2 * Dim
    temp_cell_facet_neighbors_offset = temp_topology.cell_facet_neighbors_offset
    temp_cell_facet_neighbors_length = temp_topology.cell_facet_neighbors_length
    n_neighborhoods = 0
    ref_cell_neighborhood = get_ref_cell_neighborhood(getrefshape(getcelltype(grid.base_grid)))
    for i in 1:length(grid.kopp_cells)
        new_i = refinement_cache.children_updated_indices[i]
        if refinement_cache.marked_for_refinement[new_i]
            for new_seq in 1:2^Dim
                for facet in 1:NFacets
                    n_facet_neighbors = 0
                    if ref_cell_neighborhood[new_seq][facet] == 0
                        n_inherited_neighbors = count_inherited_neighbors(i, refinement_cache, facet, temp_topology, grid, new_seq)
                        n_facet_neighbors += n_inherited_neighbors
                    else
                        n_facet_neighbors += 1
                    end
                    topology.cell_facet_neighbors_offset[facet, new_i+new_seq] = n_facet_neighbors == 0 ? 0 : n_neighborhoods + 1
                    topology.cell_facet_neighbors_length[facet, new_i+new_seq] = n_facet_neighbors == 0 ? 0 : n_facet_neighbors
                    n_neighborhoods += n_facet_neighbors
                end
            end
        else
            for facet in 1:NFacets
                n_facet_neighbors = 0
                neighbors_offset = temp_cell_facet_neighbors_offset[facet, i]
                neighbors_length = temp_cell_facet_neighbors_length[facet, i]
                for neighbor in @view temp_topology.neighbors[neighbors_offset:neighbors_offset+neighbors_length-1]
                    if refinement_cache.marked_for_refinement[refinement_cache.children_updated_indices[neighbor[1]]]
                        n_facet_neighbors += 2^(Dim - 1)
                    else
                        n_facet_neighbors += 1
                    end
                end
                topology.cell_facet_neighbors_offset[facet, new_i] = n_facet_neighbors == 0 ? 0 : n_neighborhoods + 1
                topology.cell_facet_neighbors_length[facet, new_i] = n_facet_neighbors == 0 ? 0 : n_facet_neighbors
                n_neighborhoods += n_facet_neighbors
            end
        end
    end
    return n_neighborhoods
end

function update_neighbors!(
    grid::KoppGrid{Dim},
    temp_grid::KoppGrid{Dim},
    topology::KoppTopology,
    temp_topology::KoppTopology,
    refinement_cache::KoppRefinementCache) where {Dim}
    NFacets = 2 * Dim
    temp_cell_facet_neighbors_offset = temp_topology.cell_facet_neighbors_offset
    temp_cell_facet_neighbors_length = temp_topology.cell_facet_neighbors_length
    n_neighborhoods = 0
    ref_cell_neighborhood = get_ref_cell_neighborhood(getrefshape(getcelltype(grid.base_grid)))
    for (i, cell) in enumerate(temp_grid.kopp_cells)
        new_i = refinement_cache.children_updated_indices[i]
        if refinement_cache.marked_for_refinement[new_i]
            for new_seq in 1:2^Dim
                for facet in 1:NFacets
                    n_facet_neighbors = 0
                    if ref_cell_neighborhood[new_seq][facet] == 0
                        n_inherited_neighbors = add_inherited_neighbors!(n_neighborhoods, i, refinement_cache, facet, topology, temp_topology, grid, new_seq)
                        n_facet_neighbors += n_inherited_neighbors
                    else
                        n_facet_neighbors += 1
                        topology.neighbors[n_neighborhoods + 1] = FacetIndex(new_i + ref_cell_neighborhood[new_seq][facet], get_ref_face_neighborhood(getrefshape(getcelltype(grid.base_grid)))[facet])
                    end
                    n_neighborhoods += n_facet_neighbors
                end
            end
        else
            for facet in 1:NFacets
                n_facet_neighbors = 0
                neighbors_offset = temp_cell_facet_neighbors_offset[facet, i]
                neighbors_length = temp_cell_facet_neighbors_length[facet, i]
                for neighbor in @view temp_topology.neighbors[neighbors_offset:neighbors_offset+neighbors_length-1]
                    if refinement_cache.marked_for_refinement[refinement_cache.children_updated_indices[neighbor[1]]]
                        # TODO: double check if we use old or new index here
                        for (i, new_seq) in enumerate(Ferrite.reference_facets(getrefshape(getcelltype(grid.base_grid)))[neighbor[2]])
                            topology.neighbors[n_neighborhoods + n_facet_neighbors + i] = FacetIndex(refinement_cache.children_updated_indices[neighbor[1]] + new_seq, get_ref_face_neighborhood(getrefshape(getcelltype(grid.base_grid)))[facet])
                        end
                        n_facet_neighbors += 2^(Dim - 1)

                    else
                        n_facet_neighbors += 1
                        topology.neighbors[n_neighborhoods + n_facet_neighbors] = FacetIndex(refinement_cache.children_updated_indices[neighbor[1]], neighbor[2])
                    end
                end
                topology.cell_facet_neighbors_offset[facet, new_i] = n_facet_neighbors == 0 ? 0 : n_neighborhoods + 1
                topology.cell_facet_neighbors_length[facet, new_i] = n_facet_neighbors == 0 ? 0 : n_facet_neighbors
                n_neighborhoods += n_facet_neighbors
            end
        end
    end
    return n_neighborhoods
end

function update_cells!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache) where Dim
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old cells
        grid.kopp_cells[new] =  grid.kopp_cells[old]
    end
    for (cell_idx, cell) in enumerate(grid.kopp_cells)
        refinement_cache.marked_for_refinement[cell_idx] || continue
        for new_seq in 1 : 2^Dim
            child_idx = cell_idx + new_seq
            grid.kopp_cells[child_idx] = KoppCell{Dim, Int}(cell_idx, (cell.sequence << (Dim + 1)) + new_seq)
        end
    end
end

function update_coarsened_cells!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache) where Dim
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old cells
        new == 0 && continue
        grid.kopp_cells[new] =  grid.kopp_cells[old]
    end
end

function update_root_idx!(
    grid::KoppGrid{Dim},
    topology::KoppTopology,
    temp_topology::KoppTopology,
    refinement_cache::KoppRefinementCache) where Dim
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old cells
        topology.root_idx[new] =  temp_topology.root_idx[old]
    end
    for (cell_idx, cell) in enumerate(grid.kopp_cells)
        refinement_cache.marked_for_refinement[cell_idx] || continue
        for new_seq in 1 : 2^Dim
            child_idx = cell_idx + new_seq
            topology.root_idx[child_idx] = topology.root_idx[cell_idx]
        end
    end
end

function update_coarsened_root_idx!(
    grid::KoppGrid{Dim},
    topology::KoppTopology,
    temp_topology::KoppTopology,
    refinement_cache::KoppRefinementCache) where Dim
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old cells
        new == 0 && continue
        topology.root_idx[new] =  temp_topology.root_idx[old]
    end
end

function _resize_bunch_of_stuff!(
    grid::KoppGrid{Dim},
    topology::KoppTopology,
    kopp_cache::KoppCache,
    dh::DofHandler,
    n_neighborhoods::Int,
    new_length::Int) where {Dim}
    resize!(topology.neighbors, n_neighborhoods)
    resize!(grid.kopp_cells, new_length)
    resize!(kopp_cache.cell_matrices, size(kopp_cache.cell_matrices)[1], size(kopp_cache.cell_matrices)[2], new_length)
    resize!(topology.root_idx, new_length)
    resize!(kopp_cache.interface_matrix_index, n_neighborhoods)
    resize!(kopp_cache.interface_matrices,size(kopp_cache.interface_matrices)[1], size(kopp_cache.interface_matrices)[2], n_neighborhoods ÷ 2)
    resize!(dh.cell_dofs, new_length * dh.subdofhandlers[1].ndofs_per_cell)
    resize!(dh.cell_dofs_offset, new_length)
    resize!(dh.cell_to_subdofhandler, new_length)
    resize!(kopp_cache.ansatz_isactive, new_length * dh.subdofhandlers[1].ndofs_per_cell)
    return nothing
end

# function do_the_neighbor_thing_with_sets_hehe(grid, temp_grid, topology, temp_topology, refinement_cache)
#     neighbors = Set{}()

# end

function count_inherited_neighbors(cell_idx::Int, refinement_cache::KoppRefinementCache, facet::Int, topology::KoppTopology, grid::KoppGrid{Dim}, new_seq::Int) where {Dim}
    neighbors_count = 0
    parent_facet_neighborhood_offset = topology.cell_facet_neighbors_offset[facet, cell_idx]
    parent_facet_neighborhood_length = topology.cell_facet_neighbors_length[facet, cell_idx]
    parent_facet_neighbors = @view topology.neighbors[parent_facet_neighborhood_offset:parent_facet_neighborhood_offset+parent_facet_neighborhood_length-1]
    for neighbor in parent_facet_neighbors
        neighbor_idx = neighbor[1]
        neighbor_facet = neighbor[2]
        cell = grid.kopp_cells[cell_idx]
        neighbor_cell = grid.kopp_cells[neighbor_idx]
        cell_facet_orientation_info = topology.root_facet_orientation_info[topology.root_idx[cell_idx]+facet-1]
        neighbor_facet_orientation_info = topology.root_facet_orientation_info[topology.root_idx[neighbor_idx]+neighbor_facet-1]
        cell_children = Ferrite.reference_facets(getrefshape(getcelltype(grid.base_grid)))[facet]
        neighbor_children = SVector(Ferrite.reference_facets(getrefshape(getcelltype(grid.base_grid)))[neighbor_facet])
        flip_interface = (cell_facet_orientation_info.flipped ⊻ neighbor_facet_orientation_info.flipped)
        shift = cell_facet_orientation_info.shift_index - (flip_interface ? neighbor_facet_orientation_info.shift_index : length(neighbor_children) - neighbor_facet_orientation_info.shift_index)
        new_neighbor_children = neighbor_children
        # URGERNT TODO: find a way to enable next line without allocations
        # new_neighbor_children = circshift((flip_interface ? reverse(neighbor_children) : neighbor_children), shift)
        neighbor_local_seq = neighbor_cell.sequence & (((1 << (Dim + 1)) - 1) << ((Dim + 1) * get_refinement_level(cell)))
        if refinement_cache.marked_for_refinement[refinement_cache.children_updated_indices[neighbor_idx]]
            if (get_refinement_level(neighbor_cell) > get_refinement_level(cell)) && findfirst(==(new_seq), cell_children) == findfirst(==(neighbor_local_seq), new_neighbor_children)
                neighbors_count += length(neighbor_children)
                continue
            else
                neighbors_count += 1
                continue
            end
        else
            neighbors_count += 1
            continue
        end
    end
    return neighbors_count
end

function update_dofs!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache,
    dh::DofHandler,
    temp_celldofs::Vector{Int},
    temp_cell_dofs_offset::Vector{Int},
    temp_cell_to_subdofhandler::Vector{Int}) where Dim
    dh.cell_dofs_offset .= 0
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old dofhander vectors
        dh.cell_dofs_offset[new] = temp_cell_dofs_offset[old]
        dh.cell_dofs[dh.cell_dofs_offset[new] : dh.cell_dofs_offset[new] + dh.subdofhandlers[1].ndofs_per_cell - 1] .= @view temp_celldofs[temp_cell_dofs_offset[old] : temp_cell_dofs_offset[old] + dh.subdofhandlers[1].ndofs_per_cell - 1]
        dh.cell_to_subdofhandler[new] = temp_cell_to_subdofhandler[old]
    end
    for (cell_idx, cell) in enumerate(grid.kopp_cells)
        refinement_cache.marked_for_refinement[cell_idx] || continue
        for new_seq in 1 : 2^Dim
            child_idx = cell_idx + new_seq
            # DofHandler add new dofs and offsets
            # For DG only
            dh.cell_dofs_offset[child_idx] = maximum(dh.cell_dofs_offset) + dh.subdofhandlers[1].ndofs_per_cell
            dh.cell_to_subdofhandler[child_idx] = dh.cell_to_subdofhandler[cell_idx]
            dh.cell_dofs[dh.cell_dofs_offset[child_idx] : dh.cell_dofs_offset[child_idx] + dh.subdofhandlers[1].ndofs_per_cell - 1] .= (dh.ndofs + 1) : (dh.ndofs + dh.subdofhandlers[1].ndofs_per_cell)
            dh.ndofs += dh.subdofhandlers[1].ndofs_per_cell
        end
    end
end

function update_coarsened_dofs!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache,
    dh::DofHandler,
    temp_celldofs::Vector{Int},
    temp_cell_dofs_offset::Vector{Int},
    temp_cell_to_subdofhandler::Vector{Int}) where Dim
    # dh.cell_dofs_offset .= 0
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old dofhander vectors
        new == 0 && continue
        dh.cell_dofs_offset[new] = temp_cell_dofs_offset[old]
    end
    for (cell_idx, cell) in enumerate(grid.kopp_cells)
        refinement_cache.marked_for_coarsening[cell_idx] || continue
        for (other_cell_idx, other_cell) in enumerate(grid.kopp_cells)
            refinement_cache.children_updated_indices[other_cell_idx] == 0 && continue
            dh.cell_dofs_offset[other_cell_idx] <= temp_cell_dofs_offset[cell_idx] && continue
            dh.cell_dofs_offset[other_cell_idx] -= dh.subdofhandlers[1].ndofs_per_cell * (2^Dim)
            dh.ndofs -= dh.subdofhandlers[1].ndofs_per_cell * (2^Dim)
        end
    end
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old dofhander vectors
        new == 0 && continue
        dh.cell_dofs[dh.cell_dofs_offset[new] : dh.cell_dofs_offset[new] + dh.subdofhandlers[1].ndofs_per_cell - 1] .= @view temp_celldofs[temp_cell_dofs_offset[old] : temp_cell_dofs_offset[old] + dh.subdofhandlers[1].ndofs_per_cell - 1]
        dh.cell_to_subdofhandler[new] = temp_cell_to_subdofhandler[old]
        refinement_cache.marked_for_coarsening[new] || continue
        parent_offset = dh.cell_dofs_offset[new]
        parent_sdh = dh.subdofhandlers[dh.cell_to_subdofhandler[new]]
        parent_dofs = @view dh.cell_dofs[parent_offset : parent_offset + parent_sdh.ndofs_per_cell - 1]
        for new_seq in 1 : 2^Dim
            child_idx = old + new_seq
            # For DG only
            offset = temp_cell_dofs_offset[child_idx]
            dofs = @view temp_celldofs[offset : offset + parent_sdh.ndofs_per_cell - 1]
            parent_dofs[new_seq] = dofs[new_seq]
        end
    end
end

@generated function get_child_weighted_dofs(::Type{T}) where {Dim, T <: Ferrite.RefHypercube{Dim}}
    ip = Lagrange{T, 1}()
    coords = Ferrite.reference_coordinates(ip)
    ntuple(seq -> (map!(node -> (node + coords[seq])/2, coords, coords);
    nshape_functions = length(coords);
    # TODO URGENT check if rows and cols are not switched
    SMatrix{nshape_functions, nshape_functions, Float64}(Ferrite.reference_shape_value(ip, coords[child_node], i) for i in 1:nshape_functions, child_node in 1:nshape_functions)),2^Dim)
end

function interpolate_solution!(refshape::Type{<:Ferrite.RefHypercube}, u::Vector{Float64}, dofs, parent_dofs, seq)  
    u_parent = @view u[parent_dofs]
    weights = get_child_weighted_dofs(refshape)[seq]
    for i in 1:length(dofs)
        u[dofs[i]] = u_parent ⋅ (@view weights[i,:])
    end
end
function update_koppcache!(
    grid::KoppGrid{Dim},
    refinement_cache::KoppRefinementCache,
    topology::KoppTopology,
    temp_topology::KoppTopology,
    kopp_cache::KoppCache,
    kopp_values::KoppValues,
    dh::DofHandler,
    NFacets::Int) where Dim
    for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old cells
        kopp_cache.cell_matrices[:,:,new] .= @view kopp_cache.cell_matrices[:,:,old]
        old != new && (kopp_cache.cell_matrices[:,:,new] .= 0.0) 
    end
    for cell_cache in Ferrite.CellIterator(grid, UpdateFlags(false, true, true))
        cell_idx = cell_cache.cellid
        cell = getcells(grid, cell_idx)
        # Calculate new interfaces
        if refinement_cache.marked_for_refinement[cell_idx]
            parent_offset = dh.cell_dofs_offset[cell_idx]
            parent_sdh = dh.subdofhandlers[dh.cell_to_subdofhandler[cell_idx]]
            parent_dofs = @view dh.cell_dofs[parent_offset : parent_offset + parent_sdh.ndofs_per_cell - 1]
            for new_seq in 1 : 2^Dim
                child_idx = cell_idx + new_seq
                # DofHandler add new dofs and offsets
                # For DG only
                offset = dh.cell_dofs_offset[child_idx]
                sdh = dh.subdofhandlers[dh.cell_to_subdofhandler[child_idx]]
                dofs = @view dh.cell_dofs[offset : offset + sdh.ndofs_per_cell - 1]
                interpolate_solution!(Ferrite.RefHypercube{Dim}, kopp_cache.u, dofs, parent_dofs, new_seq)
            end
        end
        if all(x -> x == 0.0, @view kopp_cache.cell_matrices[:,:,cell_idx])
            reinit!(kopp_values.cell_values, cell_cache)
            assemble_element_matrix!((@view kopp_cache.cell_matrices[:,:,cell_idx]), kopp_values)
        end
    end
    ninterfaces = 1
    facet_idx = 1
    nneighbors = 1
    interface_index = 1
    ii = Ferrite.InterfaceIterator(dh, topology)
    for (facet_idx, nneighbors, ninterfaces) in ii
        interface_cache = ii.cache
        _facet_idx = nneighbors == 1 ? facet_idx - 1 : facet_idx
        cell_idx = (_facet_idx - 1) ÷ (2*Dim) + 1
        facet_a = (_facet_idx - 1) % (2*Dim) + 1
        neighbor = FacetIndex(interface_cache.b.cc.cellid, interface_cache.b.current_facet_id)
        kopp_cache.interface_matrix_index[topology.cell_facet_neighbors_offset[facet_a, cell_idx] + nneighbors - 1] = ninterfaces
        rng  = topology.cell_facet_neighbors_offset[neighbor[2], neighbor[1]] : topology.cell_facet_neighbors_offset[neighbor[2], neighbor[1]] + topology.cell_facet_neighbors_length[neighbor[2], neighbor[1]] - 1
        # @assert any(x -> x == 0, @view kopp_cache.interface_matrix_index[rng])
        for j in rng
            kopp_cache.interface_matrix_index[j] != 0 && continue
            kopp_cache.interface_matrix_index[j] = ninterfaces
            break
        end
        reinit!(kopp_values.interface_values, interface_cache, topology)
        assemble_interface_matrix!((@view kopp_cache.interface_matrices[:,:,interface_index]), kopp_values)
        interface_index += 1
    end
end

function update_coarsened_koppcache!(
    grid::KoppGrid,
    refinement_cache::KoppRefinementCache,
    topology::KoppTopology,
    temp_topology::KoppTopology,
    kopp_cache::KoppCache,
    kopp_values::KoppValues,
    dh::DofHandler,
    temp_dh::DofHandler,
    NFacets::Int,
    Dim::Int)
    @inbounds for (old, new) in enumerate(refinement_cache.children_updated_indices)
        # copy old cells
        new == 0 && continue
        kopp_cache.cell_matrices[:,:,new] .= @view kopp_cache.cell_matrices[:,:,old]
        old != new && (kopp_cache.cell_matrices[:,:,new] .= 0.0) 
    end
    @inbounds for cell_cache in Ferrite.CellIterator(grid)
        cell_idx = cell_cache.cellid
        cell = getcells(grid, cell_idx)
        if all(x -> x == 0.0, @view kopp_cache.cell_matrices[:,:,cell_idx])
            reinit!(kopp_values.cell_values, cell_cache)
            assemble_element_matrix!((@view kopp_cache.cell_matrices[:,:,cell_idx]), kopp_values)
        end
    end
    ninterfaces = 1
    facet_idx = 1
    nneighbors = 1
    interface_index = 1
    ii = Ferrite.InterfaceIterator(dh, topology)
    for (facet_idx, nneighbors, ninterfaces) in ii
        interface_cache = ii.cache
        _facet_idx = nneighbors == 1 ? facet_idx - 1 : facet_idx
        cell_idx = (_facet_idx - 1) ÷ (2*Dim) + 1
        facet_a = (_facet_idx - 1) % (2*Dim) + 1
        neighbor = FacetIndex(interface_cache.b.cc.cellid, interface_cache.b.current_facet_id)
        kopp_cache.interface_matrix_index[topology.cell_facet_neighbors_offset[facet_a, cell_idx] + nneighbors - 1] = ninterfaces
        rng  = topology.cell_facet_neighbors_offset[neighbor[2], neighbor[1]] : topology.cell_facet_neighbors_offset[neighbor[2], neighbor[1]] + topology.cell_facet_neighbors_length[neighbor[2], neighbor[1]] - 1
        # @assert any(x -> x == 0, @view kopp_cache.interface_matrix_index[rng])
        for j in rng
            kopp_cache.interface_matrix_index[j] != 0 && continue
            kopp_cache.interface_matrix_index[j] = ninterfaces
            break
        end
        reinit!(kopp_values.interface_values, interface_cache, topology)
        assemble_interface_matrix!((@view kopp_cache.interface_matrices[:,:,interface_index]), kopp_values)
        interface_index += 1
    end
end

function add_inherited_neighbors!(n_neighborhoods::Int, cell_idx::Int, refinement_cache::KoppRefinementCache, facet::Int, topology::KoppTopology, temp_topology::KoppTopology, grid::KoppGrid{Dim}, new_seq::Int) where {Dim}
    neighbors_count = 0
    parent_facet_neighborhood_offset = temp_topology.cell_facet_neighbors_offset[facet, cell_idx]
    parent_facet_neighborhood_length = temp_topology.cell_facet_neighbors_length[facet, cell_idx]
    parent_facet_neighbors = @view temp_topology.neighbors[parent_facet_neighborhood_offset:parent_facet_neighborhood_offset+parent_facet_neighborhood_length-1]
    for neighbor in parent_facet_neighbors
        neighbor_idx = neighbor[1]
        neighbor_facet = neighbor[2]
        cell = grid.kopp_cells[cell_idx]
        neighbor_cell = grid.kopp_cells[refinement_cache.children_updated_indices[neighbor_idx]]
        cell_facet_orientation_info = temp_topology.root_facet_orientation_info[temp_topology.root_idx[cell_idx]+facet-1]
        neighbor_facet_orientation_info = temp_topology.root_facet_orientation_info[temp_topology.root_idx[neighbor_idx]+neighbor_facet-1]
        cell_children = Ferrite.reference_facets(getrefshape(getcelltype(grid.base_grid)))[facet]
        neighbor_children = SVector(Ferrite.reference_facets(getrefshape(getcelltype(grid.base_grid)))[neighbor_facet])
        flip_interface = (cell_facet_orientation_info.flipped ⊻ neighbor_facet_orientation_info.flipped)
        shift = cell_facet_orientation_info.shift_index - (flip_interface ? neighbor_facet_orientation_info.shift_index : length(neighbor_children) - neighbor_facet_orientation_info.shift_index)
        new_neighbor_children = neighbor_children
        # URGERNT TODO: find a way to enable next line without allocations
        # new_neighbor_children = circshift((flip_interface ? reverse(neighbor_children) : neighbor_children), shift)
        neighbor_local_seq = neighbor_cell.sequence & (((1 << (Dim + 1)) - 1) << ((Dim + 1) * get_refinement_level(cell)))
        if refinement_cache.marked_for_refinement[refinement_cache.children_updated_indices[neighbor_idx]]
            if (get_refinement_level(neighbor_cell) > get_refinement_level(cell)) && findfirst(==(new_seq), cell_children) == findfirst(==(neighbor_local_seq), new_neighbor_children)
                topology.neighbors[n_neighborhoods + neighbors_count + 1 : n_neighborhoods + neighbors_count + length(neighbor_children)] = [FacetIndex(refinement_cache.children_updated_indices[neighbor_idx] + neighbor_child, neighbor_facet) for neighbor_child in neighbor_children]
                neighbors_count += length(neighbor_children)
                continue
            else
                topology.neighbors[n_neighborhoods + neighbors_count + 1] = FacetIndex(refinement_cache.children_updated_indices[neighbor_idx] + new_neighbor_children[findfirst(==(new_seq), cell_children)], neighbor_facet)
                neighbors_count += 1
                continue
            end
        else
            topology.neighbors[n_neighborhoods + neighbors_count + 1] = FacetIndex(refinement_cache.children_updated_indices[neighbor_idx], neighbor_facet)
            neighbors_count += 1
            continue
        end
    end
    return neighbors_count
end

@generated function get_ref_face_neighborhood(::Type{T}) where {Dim, T <: Ferrite.RefHypercube{Dim}}
    I = Tensors.diagm(Tensor{2,Dim}, ones(Dim))
    normals = Ferrite.weighted_normal.(Ref(I), T, 1:2*Dim)
    return ntuple(facet -> findfirst(x -> x ≈ -Ferrite.weighted_normal(I, T, facet), normals) , Val(Dim * 2))
end

# 1 alloc for hexahedron only IDK why hehe
@generated function get_ref_cell_neighborhood(::Type{T}) where {Dim, T <: Ferrite.RefHypercube{Dim}}
    return ntuple(cell -> 
        ntuple(facet -> cell ∈ Ferrite.reference_facets(T)[facet] ? 0 : 
            Ferrite.reference_facets(T)[facet][findfirst(facet_node -> any(facet_node .∈ filter(nodes ->  cell ∈ nodes, Ferrite.reference_edges(T))), Ferrite.reference_facets(T)[facet])], Val(Dim * 2)),
        Val(2^Dim))
end

function _process_seq(coords::NTuple{NNodes, NodeType}, seq::TInt) where {NodeType, TInt, NNodes}
    Dim = TInt(log2(NNodes))
    level::TInt = 0
    nbits::TInt = Dim + 1
    mask = (1 << nbits) - 1 # Does this need to be T too?
    maximum_level::TInt = sizeof(TInt)*8 ÷ nbits # Maybe use ÷ ?
    coord_new = MVector{NNodes, NodeType}(coords)
    coord_new_temp = MVector{NNodes, NodeType}(coords)
    @inbounds for level::TInt in maximum_level:-1:0 # There should be an easier way
        local_seq::TInt = (seq & (mask << (nbits * level)) ) >> (nbits * level )
        local_seq == 0 && continue
        @inbounds for (i, coord) in enumerate(coord_new_temp) 
            coord_new[i] = NodeType((coord.x + coord_new_temp[local_seq].x)/2)
        end
        coord_new_temp .= coord_new
    end
    return Tuple(coord_new)
end

function _transform_to_parent!(coords::AbstractVector, node_coords, seq::T) where {T}
    Dim = 2
    level::T = 0
    nbits::T = Dim + 1
    mask = (1 << nbits) - 1 # Does this need to be T too?
    maximum_level::T = sizeof(T)*8 ÷ nbits # Maybe use ÷ ?
    # coord_new = MVector{length(coords), eltype(coords)}(coords)
    for level::T in maximum_level:0 # There should be an easier way
        local_seq = seq & (mask << (nbits * level))
        local_seq == 0 && continue
        map!(node -> ((node + node_coords[local_seq])/2), coords, coords)
    end
    return nothing
end

function get_root_idx(grid::KoppGrid, i::Int)
    parent = grid.kopp_cells[i].parent
    # TODO: URGENT maxiter
    while parent > 0
        parent = grid.kopp_cells[parent].parent
    end
    return -parent
end

# Works only for 1st order geometric interpolation
# Has 0 allocations
function get_refined_coords(grid::KoppGrid{Dim}, i::Int) where Dim
    cell = grid.kopp_cells[i]
    root_idx = get_root_idx(grid, i)
    node_ids = Ferrite.get_node_ids(grid.base_grid.cells[root_idx])
    root_coords = ntuple(i -> ( grid.base_grid.nodes[node_ids[i]]), Val(2^(Dim)))
    return _process_seq(root_coords, cell.sequence)
end

# # 5 allocs
# function get_refinement_face_neighborhood(grid::KoppGrid{Dim}, topology::KoppTopology, i::Int) where {Dim}
#     cell = grid.kopp_cells[i]
#     seq = cell.sequence
#     refshape = getrefshape(getcelltype(grid.base_grid))
#     ref_neighborhood = get_ref_face_neighborhood(refshape)
#     mask = (1 << (Dim + 1)) - 1 # Does this need to be T too?
#     local_seq = seq & mask 
#     inherit_from_parent = local_seq .∈ Ferrite.reference_facets(refshape)
#     parent_neighbors = topology.neighbors[cell.parent]
#     return ntuple(facet -> inherit_from_parent[facet] ? copy(parent_neighbors[facet]) : [ref_neighborhood[facet]] , Val(Dim * 2))
# end

# @generated function get_children_cells_from_facet(::Type{T}) where {Dim, T <: Ferrite.RefHypercube{Dim}}
#     return Ferrite.reference_facets(T)
# end

function Ferrite.getcoordinates!(dst::Vector{Vec{dim,T}}, grid::KoppGrid, cell::Int) where{dim, T}
    for i in 1:length(dst)
        dst[i] =  get_refined_coords(grid, cell)[i].x
    end
end

function Ferrite.getcoordinates!(dst::Vector{Vec{dim,T}}, grid::KoppGrid, cell::CellIndex) where{dim, T}
    return Ferrite.getcoordinates!(dst, grid, cell.idx)
end
