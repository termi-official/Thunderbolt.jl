# Generates a hexahedral ring in the form of a large cylinder subtracted by a smaller cylinder.
function generate_ring_mesh(ne_c, ne_r, ne_l; radial_inner::T = Float64(0.75), radial_outer::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = ne_c*ne_r*ne_l;
    n_nodes_c = ne_c; n_nodes_r = ne_r+1; n_nodes_l = ne_l+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    radial_coords = range(radial_inner, stop=radial_outer, length=n_nodes_r)
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
    for k in 1:ne_l, j in 1:ne_r, i in 1:ne_c
        i_next = (i == ne_c) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(ne_c, ne_r, ne_l))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            #[FaceIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            #[FaceIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["Myocardium"] = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facesets["Endocardium"]  = Set{FaceIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facesets["Epicardium"]   = Set{FaceIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facesets["Base"]    = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    return Grid(cells, nodes, facesets=facesets)
end

# Generates a hexahedral truncated ellipsoidal mesh by reparametrizing a hollow sphere (r=1.0 length units) where longitudinal_upper determines the truncation height.
"""
    generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinally::Int; inner_chamber_radius = 0.7, outer_wall_radius = 1.0, longitudinal_cutoff_lower::T = Float64(-1.0), longitudinal_cutoff_upper::T = Float64(0.2), longitudinal_stretch::T = Float64(1.2))

Generate an idealized left ventricle as a truncated ellipsoid.
The number of elements per axis are controlled by the 3 parameters.
"""
function generate_ideal_lv_mesh(ne_c::Int, ne_r::Int, ne_l::Int; radial_inner::T = Float64(0.7), radial_outer::T = Float64(1.0), longitudinal_upper::T = Float64(0.2), apex_inner::T = Float64(1.3), apex_outer::T = Float64(1.5)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = ne_c*ne_r*ne_l;
    n_nodes_c = ne_c; n_nodes_r = ne_r+1; n_nodes_l = ne_l+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    # Take a ring section of the heart and mark its circumferential coordinate by its angle
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    # Untransformed radial coordinate of a ring section
    radii_in_percent = range(0.0, stop=1.0, length=n_nodes_r)
    # z axis expressed as the angle between the apicobasal vector and the current layer from apex (0.0) to base ((1.0+longitudinal_upper)*π/2)
    longitudinal_angle = range(0, stop=(1.0+longitudinal_upper)*π/2, length=n_nodes_l+1)
    nodes = Node{3,T}[]
    # Add everything but apex and base
    for θ ∈ longitudinal_angle[2:(end-1)], radius_percent ∈ radii_in_percent, φ ∈ circumferential_angle[1:(end-1)]
        # thickness_inner = radial_inner*longitudinal_percent + apex_inner*(1.0-longitudinal_percent)
        # thickness_outer = radial_outer*longitudinal_percent + apex_outer*(1.0-longitudinal_percent)
        # radius = radius_percent*thickness_outer + (1.0-radius_percent)*thickness_inner
        radius = radial_inner*(1.0-radius_percent) + radial_outer*(radius_percent)
        # cylindrical -> carthesian
        z = θ < π/2 ? (apex_inner*(1.0-radius_percent) + apex_outer*(radius_percent))*cos(θ) : apex_outer*cos(θ)
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), z)))
    end

    # Add flat base
    for θ ∈ longitudinal_angle[end], radius_percent ∈ radii_in_percent, φ ∈ circumferential_angle[1:(end-1)]
        radius = radial_inner*(1.0-radius_percent) + radial_outer*(radius_percent)
        # cylindrical -> carthesian
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), apex_outer*cos(θ))))
    end

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Union{Hexahedron,Wedge,Triangle}[]
    for k in 1:ne_l, j in 1:ne_r, i in 1:ne_c
        i_next = (i == ne_c) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(ne_c, ne_r, ne_l))
    boundary = FaceIndex[[FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                          [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                          [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # boundary_matrix = boundaries_to_sparse(boundary)

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
    for j ∈ 1:ne_r, i ∈ 1:ne_c
        i_next = (i == ne_c) ? 1 : i + 1
        singular_index = length(nodes)-ne_r+j-1
        push!(cells, Wedge((
            singular_index , node_array[i,j,1], node_array[i_next,j,1],
            singular_index+1, node_array[i,j+1,1], node_array[i_next,j+1,1],
        )))
    end

    # return Grid(cells, nodes, nodesets=nodesets, facesets=facesets)
    return Grid(cells, nodes, facesets=facesets)
end

# Generates a hexahedral truncated ellipsoidal mesh by reparametrizing a hollow sphere (r=1.0 length units) where longitudinal_upper determines the truncation height.
function generate_ideal_lv_mesh_boxed(ne_c, ne_r, ne_l; radial_inner::T = Float64(0.75), radial_outer::T = Float64(1.0), longitudinal_lower::T = Float64(-1.0), longitudinal_upper::T = Float64(0.2)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = ne_c*ne_r*ne_l;
    n_nodes_c = ne_c; n_nodes_r = ne_r+1; n_nodes_l = ne_l+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    generator_offset = 5

    # Generate nodes
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    radial_coords = range(radial_inner, stop=radial_outer, length=n_nodes_r)
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
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), radial_outer*cos(θ))))
    end

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k in 1:ne_l, j in 1:ne_r, i in 1:ne_c
        i_next = (i == ne_c) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(ne_c, ne_r, ne_l))
    boundary = FaceIndex[[FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                          [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                          [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # boundary_matrix = boundaries_to_sparse(boundary)

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
    coords_x = range(minx, stop=maxx, length=Int(ne_c/4)+1)
    coords_y = range(miny, stop=maxy, length=Int(ne_c/4)+1)
    coords_z = radial_coords#range(minz, stop=maxz, length=ne_r+1)
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
    for k in 1:(ne_r), j in 1:(Int(ne_c/4)), i in 1:(Int(ne_c/4))
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
