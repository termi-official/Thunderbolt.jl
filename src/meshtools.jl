"""
    generate_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
"""
function generate_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    radial_coords = range(inner_radius, stop=outer_radius, length=n_nodes_r)
    longitudinal_angle = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = radial_coords[j]-apicobasal_tilt*longitudinal_angle[k]/maximum(abs.(longitudinal_angle))
        push!(nodes, Node((radius*cos(circumferential_angle[i]), radius*sin(circumferential_angle[i]), longitudinal_angle[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k in 1:num_elements_logintudinal, j in 1:num_elements_radial, i in 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            #[FaceIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            #[FaceIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["Myocardium"] = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facesets["Endocardium"]  = Set{FaceIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facesets["Epicardium"]   = Set{FaceIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facesets["Base"]    = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    return Grid(cells, nodes, facesets=facesets)
end

"""
    generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinally::Int; inner_chamber_radius = 0.7, outer_wall_radius = 1.0, longitudinal_cutoff_lower::T = Float64(-1.0), longitudinal_cutoff_upper::T = Float64(0.2), longitudinal_stretch::T = Float64(1.2))

Generate an idealized left ventricle as a truncated ellipsoid.
The number of elements per axis are controlled by the first three parameters.
"""
function generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.7), outer_radius::T = Float64(1.0), longitudinal_upper::T = Float64(0.2), apex_inner::T = Float64(1.3), apex_outer::T = Float64(1.5)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential; n_nodes_r = num_elements_radial+1; n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    # Take a ring section of the heart and mark its circumferential coordinate by its angle
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    # Untransformed radial coordinate of a ring section
    radii_in_percent = range(0.0, stop=1.0, length=n_nodes_r)
    # z axis expressed as the angle between the apicobasal vector and the current layer from apex (0.0) to base ((1.0+longitudinal_upper)*π/2)
    longitudinal_angle = range(0, stop=(1.0+longitudinal_upper)*π/2, length=n_nodes_l+1)
    nodes = Node{3,T}[]
    # Add nodes up to apex
    for θ ∈ longitudinal_angle[2:(end-1)], radius_percent ∈ radii_in_percent, φ ∈ circumferential_angle[1:(end-1)]
        # thickness_inner = inner_radius*longitudinal_percent + apex_inner*(1.0-longitudinal_percent)
        # thickness_outer = outer_radius*longitudinal_percent + apex_outer*(1.0-longitudinal_percent)
        # radius = radius_percent*thickness_outer + (1.0-radius_percent)*thickness_inner
        radius = inner_radius*(1.0-radius_percent) + outer_radius*(radius_percent)
        # cylindrical -> carthesian
        z = θ < π/2 ? (apex_inner*(1.0-radius_percent) + apex_outer*(radius_percent))*cos(θ) : apex_outer*cos(θ)
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), z)))
    end

    # Add flat base
    for θ ∈ longitudinal_angle[end], radius_percent ∈ radii_in_percent, φ ∈ circumferential_angle[1:(end-1)]
        radius = inner_radius*(1.0-radius_percent) + outer_radius*(radius_percent)
        # cylindrical -> carthesian
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), apex_outer*cos(θ))))
    end

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Union{Hexahedron,Wedge}[]
    for k in 1:num_elements_logintudinal, j in 1:num_elements_radial, i in 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FaceIndex[[FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                          [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                          [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["Endocardium"]  = Set{FaceIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facesets["Epicardium"]   = Set{FaceIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facesets["Base"]    = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    # Add apex nodes
    for radius_percent ∈ radii_in_percent
        radius = apex_inner*(1.0-radius_percent) + apex_outer*(radius_percent)
        push!(nodes, Node((0.0, 0.0, radius)))
    end

    # Add apex cells
    for j ∈ 1:num_elements_radial, i ∈ 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        singular_index = length(nodes)-num_elements_radial+j-1
        push!(cells, Wedge((
            singular_index , node_array[i,j,1], node_array[i_next,j,1],
            singular_index+1, node_array[i,j+1,1], node_array[i_next,j+1,1],
        )))
    end

    # return Grid(cells, nodes, nodesets=nodesets, facesets=facesets)
    return Grid(cells, nodes, facesets=facesets)
end

function hexahedralize(grid::Grid{3, Hexahedron})
    return grid
end

# TODO nonlinear version
function create_center_node(grid::AbstractGrid{dim}, cell::LinearCellGeometry) where {dim}
    center = zero(Vec{dim})
    vs = vertices(cell)
    for v ∈ vs
        node = getnodes(grid, v)
        center += node.x / length(vs)
    end
    return Node(center)
end

function create_edge_center_node(grid::AbstractGrid{dim}, cell::LinearCellGeometry, edge_idx::Int) where {dim}
    center = zero(Vec{dim})
    es = edges(cell)
    for v ∈ es[edge_idx]
        node = getnodes(grid, v)
        center += node.x / length(es[edge_idx])
    end
    return Node(center)
end

function create_face_center_node(grid::AbstractGrid{dim}, cell::LinearCellGeometry, face_idx::Int) where {dim}
    center = zero(Vec{dim})
    fs = faces(cell)
    for v ∈ fs[face_idx]
        node = getnodes(grid, v)
        center += node.x / length(fs[face_idx])
    end
    return Node(center)
end

function refinum_elements_circumferentialell_uniform(mgrid::SimpleMesh3D, cell::Hexahedron, cell_idx::Int, global_edge_indices, global_face_indices)
    # Compute offsets
    new_edge_offset = num_nodes(mgrid)
    new_face_offset = num_edges(mgrid) + new_edge_offset
    # Compute indices
    vnids = vertices(cell)
    enids = new_edge_offset .+ global_edge_indices
    fnids = new_face_offset .+ global_face_indices
    cnid  = new_face_offset  + num_faces(mgrid) + cell_idx
    # Construct 8 finer cells
    return [
        Hexahedron((
            vnids[1], enids[1], fnids[1], enids[4],
            enids[9], fnids[2], cnid    , fnids[5] ,
        )),
        Hexahedron((
            enids[1], vnids[2] , enids[2], fnids[1],
            fnids[2], enids[10], fnids[3], cnid    ,
        )),
        Hexahedron((
            enids[4], fnids[1], enids[3], vnids[4],
            fnids[5], cnid    , fnids[4], enids[12],
        )),
        Hexahedron((
            fnids[1], enids[2], vnids[3] , enids[3],
            cnid    , fnids[3], enids[11], fnids[4],
        )),
        Hexahedron((
            enids[9], fnids[2], cnid    , fnids[5],
            vnids[5], enids[5], fnids[6], enids[8],
        )),
        Hexahedron((
            fnids[2], enids[10], fnids[3], cnid   ,
            enids[5], vnids[6] , enids[6], fnids[6],
        )),
        Hexahedron((
            fnids[5], cnid    , fnids[4], enids[12],
            enids[8], fnids[6], enids[7], vnids[8] ,
        )),
        Hexahedron((
            cnid    , fnids[3], enids[11], fnids[4],
            fnids[6], enids[6], vnids[7] , enids[7] 
        )),
    ]
end

# Hex into 8 hexahedra
hexahedralize_cell(mgrid::SimpleMesh3D, cell::Hexahedron, cell_idx::Int, global_edge_indices, global_face_indices) = refinum_elements_circumferentialell_uniform(mgrid, cell, cell_idx, global_edge_indices, global_face_indices)

function hexahedralize_cell(mgrid::SimpleMesh3D, cell::Wedge, cell_idx::Int, global_edge_indices, global_face_indices)
    # Compute offsets
    new_edge_offset = num_nodes(mgrid)
    new_face_offset = new_edge_offset+num_edges(mgrid)
    # Compute indices
    vnids = vertices(cell)
    enids = new_edge_offset .+ global_edge_indices
    fnids = new_face_offset .+ global_face_indices
    cnid  = new_face_offset  + num_faces(mgrid) + cell_idx
    return [
        # Bottom 3
        Hexahedron((
            vnids[1], enids[1], fnids[1], enids[2],
            enids[3], fnids[2], cnid    , fnids[3],
        )),
        Hexahedron((
            enids[1], vnids[2], enids[4], fnids[1],
            fnids[2], enids[5], fnids[4], cnid    ,
        )),
        Hexahedron((
            fnids[1], enids[4], vnids[3], enids[2],
            cnid    , fnids[4], enids[6], fnids[3],
        )),
        # Top 3
        Hexahedron((
            enids[3], fnids[2], cnid    , fnids[3],
            vnids[4], enids[7], fnids[5], enids[8],
        )),
        Hexahedron((
            fnids[2], enids[5], fnids[4], cnid    ,
            enids[7], vnids[5], enids[9], fnids[5],
        )),
        Hexahedron((
            cnid    , fnids[4], enids[6], fnids[3],
            fnids[5], enids[9], vnids[6], enids[8],
        ))
    ]
end

function uniform_refinement(grid::Grid{3,C,T}) where {C,T}
    mgrid = SimpleMesh3D(grid) # Helper

    cells = getcells(grid)

    nfacenods = length(mgrid.mfaces)
    new_face_nodes = Array{Node{3,T}}(undef, nfacenods) # We have to add 1 center node per face
    nedgenodes = length(mgrid.medges)
    new_edge_nodes = Array{Node{3,T}}(undef, nedgenodes) # We have to add 1 center node per edge
    ncellnodes = length(cells)
    new_cell_nodes = Array{Node{3,T}}(undef, ncellnodes) # We have to add 1 center node per volume

    new_cells = AbstractCell[]

    for (cellidx,cell) ∈ enumerate(cells)
        # Cell center node
        new_cell_nodes[cellidx] = create_center_node(grid, cell)
        global_edge_indices = global_edges(mgrid, cell)
        # Edge center nodes
        for (edgeidx,gei) ∈ enumerate(global_edge_indices)
            new_edge_nodes[gei] = create_edge_center_node(grid, cell, edgeidx)
        end
        # Face center nodes
        global_face_indices = global_faces(mgrid, cell)
        for (faceidx,gfi) ∈ enumerate(global_face_indices)
            new_face_nodes[gfi] = create_face_center_node(grid, cell, faceidx)
        end
        append!(new_cells, refinum_elements_circumferentialell_uniform(mgrid, cell, cellidx, global_edge_indices, global_face_indices))
    end
    # TODO boundary sets
    return Grid(new_cells, [grid.nodes; new_edge_nodes; new_face_nodes; new_cell_nodes])
end

function hexahedralize(grid::Grid{3,C,T}) where {C,T}
    mgrid = SimpleMesh3D(grid) # Helper

    cells = getcells(grid)

    nfacenods = length(mgrid.mfaces)
    new_face_nodes = Array{Node{3,T}}(undef, nfacenods) # We have to add 1 center node per face
    nedgenodes = length(mgrid.medges)
    new_edge_nodes = Array{Node{3,T}}(undef, nedgenodes) # We have to add 1 center node per edge
    ncellnodes = length(cells)
    new_cell_nodes = Array{Node{3,T}}(undef, ncellnodes) # We have to add 1 center node per volume

    new_cells = Hexahedron[]

    for (cellidx,cell) ∈ enumerate(cells)
        # Cell center node
        new_cell_nodes[cellidx] = create_center_node(grid, cell)
        global_edge_indices = global_edges(mgrid, cell)
        # Edge center nodes
        for (edgeidx,gei) ∈ enumerate(global_edge_indices)
            new_edge_nodes[gei] = create_edge_center_node(grid, cell, edgeidx)
        end
        # Face center nodes
        global_face_indices = global_faces(mgrid, cell)
        for (faceidx,gfi) ∈ enumerate(global_face_indices)
            new_face_nodes[gfi] = create_face_center_node(grid, cell, faceidx)
        end
        append!(new_cells, hexahedralize_cell(mgrid, cell, cellidx, global_edge_indices, global_face_indices))
    end
    # TODO boundary sets
    return Grid(new_cells, [grid.nodes; new_edge_nodes; new_face_nodes; new_cell_nodes])
end

# Generates a hexahedral truncated ellipsoidal mesh by reparametrizing a hollow sphere (r=1.0 length units) where longitudinal_upper determines the truncation height.
function generate_ideal_lv_mesh_boxed(num_elements_circumferential, num_elements_radial, num_elements_logintudinal; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-1.0), longitudinal_upper::T = Float64(0.2)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential; n_nodes_r = num_elements_radial+1; n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    generator_offset = 5

    # Generate nodes
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    radial_coords = range(inner_radius, stop=outer_radius, length=n_nodes_r)
    longitudinal_angle = range(0, stop=(1.0+longitudinal_upper)*π/2, length=n_nodes_l+generator_offset-1)
    nodes = Node{3,T}[]
    # Add everything but apex and base
    for θ ∈ longitudinal_angle[generator_offset:(end-1)], radius ∈ radial_coords, φ ∈ circumferential_angle[1:(end-1)]
        # cylindrical -> carthesian
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), radius*cos(θ))))
    end

    # Add flat base
    for θ ∈ longitudinal_angle[end], radius ∈ radial_coords, φ ∈ circumferential_angle[1:(end-1)]
        # cylindrical -> carthesian
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), outer_radius*cos(θ))))
    end

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k in 1:num_elements_logintudinal, j in 1:num_elements_radial, i in 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FaceIndex[[FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                          [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                          [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["Endocardium"]  = Set{FaceIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facesets["Epicardium"]   = Set{FaceIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facesets["Base"]    = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    apex_node_offset = length(nodes)

    maxx = minx = nodes[node_array[1,1,1]].x[1]
    maxy = miny = nodes[node_array[1,1,1]].x[2]
    maxz = minz = nodes[node_array[1,1,1]].x[3]
    for n ∈ nodes[node_array[:,:,1]]
        maxx = max(maxx, n.x[1])
        minx = min(minx, n.x[1])
        maxy = max(maxy, n.x[2])
        miny = min(miny, n.x[2])
        maxz = max(maxz, n.x[3])
        minz = min(minz, n.x[3])
    end

    # Making the connection continuous will be painful.
    coords_x = range(minx, stop=maxx, length=Int(num_elements_circumferential/4)+1)
    coords_y = range(miny, stop=maxy, length=Int(num_elements_circumferential/4)+1)
    coords_z = radial_coords#range(minz, stop=maxz, length=num_elements_radial+1)
    for z ∈ coords_z, y ∈ coords_y, x ∈ coords_x
        # rebox coordinate
        x = x/2.0
        y = y/2.0
        r = sqrt(x*x+y*y+z*z)
        cosθ = z/r
        push!(nodes, Node((x,y,z*cosθ)))
    end

    n_nodes = length(coords_x)*length(coords_y)*length(coords_z)
    node_array_apex = reshape(collect(1:n_nodes), (length(coords_x), length(coords_y), length(coords_z)))
    node_array_apex .+= apex_node_offset
    for k in 1:(num_elements_radial), j in 1:(Int(num_elements_circumferential/4)), i in 1:(Int(num_elements_circumferential/4))
        push!(cells, Hexahedron((node_array_apex[i,j,k], node_array_apex[i+1,j,k], node_array_apex[i+1,j+1,k], node_array_apex[i,j+1,k],
                                 node_array_apex[i,j,k+1], node_array_apex[i+1,j,k+1], node_array_apex[i+1,j+1,k+1], node_array_apex[i,j+1,k+1])))
    end

    return Grid(cells, nodes, facesets=facesets)
end

function compute_minΔx(grid::Grid{dim, CT, DT}) where {dim, CT, DT}
    Δx = DT[DT(Inf) for _ ∈ 1:getncells(grid)]
    for (cell_idx,cell) ∈ enumerate(getcells(grid)) # todo cell iterator
        for (node_idx,node1) ∈ enumerate(cell.nodes) # todo node accessor
            for node2 ∈ cell.nodes[node_idx+1:end] # nodo node accessor
                Δx[cell_idx] = min(Δx[cell_idx], norm(grid.nodes[node1].x - grid.nodes[node2].x))
            end
        end
    end
    return Δx
end

function compute_maxΔx(grid::Grid{dim, CT, DT}) where {dim, CT, DT}
    Δx = DT[DT(0.0) for _ ∈ 1:getncells(grid)]
    for (cell_idx,cell) ∈ enumerate(getcells(grid)) # todo cell iterator
        for (node1, node2) ∈ edges(cell)
            Δx[cell_idx] = max(Δx[cell_idx], norm(grid.nodes[node1].x - grid.nodes[node2].x))
        end
    end
    return Δx
end

function compute_degeneracy(grid::Grid{dim, CT, DT}) where {dim, CT, DT}
    ratio = DT[DT(0.0) for _ ∈ 1:getncells(grid)]
    for (cell_idx,cell) ∈ enumerate(getcells(grid)) # todo cell iterator
        Δxmin = DT(Inf)
        Δxmax = zero(DT)
        for (node_idx,node1) ∈ enumerate(cell.nodes) # todo node accessor
            for node2 ∈ cell.nodes[node_idx+1:end] # nodo node accessor
                Δ = norm(grid.nodes[node1].x - grid.nodes[node2].x)
                Δxmin = min(Δxmin, Δ)
                Δxmax = max(Δxmax, Δ)
            end
        end
        ratio[cell_idx] = max(ratio[cell_idx], Δxmin/Δxmax)
    end
    return ratio
end
