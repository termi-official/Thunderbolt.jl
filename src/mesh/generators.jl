"""
    generate_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with linear ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
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

    nodesets = Dict{String,Set{Int}}()
    nodesets["MyocardialAnchor1"] = Set{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = Set{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = Set{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = Set{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return Grid(cells, nodes, facesets=facesets, nodesets=nodesets)
end


"""
    generate_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with given opening angle and linear ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
The ring is opened along the Cartesian x-z plane.
"""
function generate_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential+1;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(opening_angle/2, stop=2*π-opening_angle/2, length=n_nodes_c)
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
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i + 1,j,k], node_array[i + 1,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i + 1,j,k+1], node_array[i + 1,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            [FaceIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            [FaceIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()

    facesets["Myocardium"] = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facesets["Endocardium"]  = Set{FaceIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facesets["Open1"]  = Set{FaceIndex}(boundary[(1:length(cell_array[end,:,:][:])) .+ offset]); offset += length(cell_array[end,:,:][:])
    facesets["Epicardium"]   = Set{FaceIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facesets["Open2"]   = Set{FaceIndex}(boundary[(1:length(cell_array[1,:,:][:]))   .+ offset]); offset += length(cell_array[1,:,:][:])
    facesets["Base"]    = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    nodesets = Dict{String,Set{Int}}()
    nodesets["MyocardialAnchor1"] = Set{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = Set{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = Set{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = Set{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return Grid(cells, nodes, facesets=facesets, nodesets=nodesets)
end


# const linear_index_to_local_index_table_hex27 = [1,9,2, 12,21,10, 4,11,3,  17,22,18, 25,27,23, 20,24,19, 5,13,6, 16,26,14, 8,15,7]
# const local_index_to_linear_index_table_hex27 = invperm(linear_index_to_local_index_table_hex27)
# const tensorproduct_index_to_local_index_table_hex27 = reshape(raw_index_to_local_index_table_hex27, (3,3,3))

"""
    generate_quadratic_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with quadratic ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
"""
function generate_quadratic_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = 2*num_elements_circumferential;
    n_nodes_r = 2*num_elements_radial+1;
    n_nodes_l = 2*num_elements_logintudinal+1;
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
    cells = QuadraticHexahedron[]
    for k_ in 1:num_elements_logintudinal, j_ in 1:num_elements_radial, i_ in 1:num_elements_circumferential
        i_next = (i_ == num_elements_circumferential) ? 1 : 2*i_ + 1
        i = 2*i_-1
        j = 2*j_-1
        k = 2*k_-1
        push!(cells, QuadraticHexahedron((
            node_array[i+0,j+0,k+0], node_array[i_next,j+0,k+0], node_array[i_next,j+2,k+0], node_array[i+0,j+2,k+0], # Vertex loop back
            node_array[i+0,j+0,k+2], node_array[i_next,j+0,k+2], node_array[i_next,j+2,k+2], node_array[i+0,j+2,k+2],  # Vertex loop front
            node_array[i+1,j+0,k+0], node_array[i_next,j+1,k+0], node_array[i+1,j+2,k+0],    node_array[i+0,j+1,k+0], # Edge loop back
            node_array[i+1,j+0,k+2], node_array[i_next,j+1,k+2], node_array[i+1,j+2,k+2],    node_array[i+0,j+1,k+2], # Edge loop front
            node_array[i+0,j+0,k+1], node_array[i_next,j+0,k+1], node_array[i_next,j+2,k+1], node_array[i+0,j+2,k+1], # Edge loop center
            node_array[i+1,j+1,k+0], node_array[i+1,j+0,k+1],    node_array[i_next,j+1,k+1], node_array[i+1,j+2,k+1], node_array[i+0,j+1,k+1], node_array[i+1,j+1,k+2], # Face centers
            node_array[i+1,j+1,k+1]# Center
        )))
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

    nodesets = Dict{String,Set{Int}}()
    nodesets["MyocardialAnchor1"] = Set{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = Set{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = Set{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = Set{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return Grid(cells, nodes, facesets=facesets, nodesets=nodesets)
end


"""
    generate_quadratic_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with given opening angle and quadratic ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
The ring is opened along the Cartesian x-z plane.
"""
function generate_quadratic_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = 2*num_elements_circumferential+1;
    n_nodes_r = 2*num_elements_radial+1;
    n_nodes_l = 2*num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(opening_angle/2, stop=2*π-opening_angle/2, length=n_nodes_c)
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
    cells = QuadraticHexahedron[]
    for k_ in 1:num_elements_logintudinal, j_ in 1:num_elements_radial, i_ in 1:num_elements_circumferential
        i = 2*i_-1
        j = 2*j_-1
        k = 2*k_-1
        push!(cells, QuadraticHexahedron((
            node_array[i+0,j+0,k+0], node_array[2*i_ + 1,j+0,k+0], node_array[2*i_ + 1,j+2,k+0], node_array[i+0,j+2,k+0], # Vertex loop back
            node_array[i+0,j+0,k+2], node_array[2*i_ + 1,j+0,k+2], node_array[2*i_ + 1,j+2,k+2], node_array[i+0,j+2,k+2],  # Vertex loop front
            node_array[i+1,j+0,k+0], node_array[2*i_ + 1,j+1,k+0], node_array[i+1,j+2,k+0],    node_array[i+0,j+1,k+0], # Edge loop back
            node_array[i+1,j+0,k+2], node_array[2*i_ + 1,j+1,k+2], node_array[i+1,j+2,k+2],    node_array[i+0,j+1,k+2], # Edge loop front
            node_array[i+0,j+0,k+1], node_array[2*i_ + 1,j+0,k+1], node_array[2*i_ + 1,j+2,k+1], node_array[i+0,j+2,k+1], # Edge loop center
            node_array[i+1,j+1,k+0], node_array[i+1,j+0,k+1],    node_array[2*i_ + 1,j+1,k+1], node_array[i+1,j+2,k+1], node_array[i+0,j+1,k+1], node_array[i+1,j+1,k+2], # Face centers
            node_array[i+1,j+1,k+1]# Center
        )))
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

    nodesets = Dict{String,Set{Int}}()
    nodesets["MyocardialAnchor1"] = Set{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = Set{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = Set{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = Set{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return Grid(cells, nodes, facesets=facesets, nodesets=nodesets)
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

    nodesets = Dict{String,Set{Int}}()
    nodesets["Apex"] = Set{Int}()

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
        j == 1 && push!(facesets["Endocardium"], FaceIndex(length(cells), 1))
        j == num_elements_radial && push!(facesets["Epicardium"], FaceIndex(length(cells), 5))
        j == num_elements_radial && push!(nodesets["Apex"], singular_index+1)
    end

    return Grid(cells, nodes, nodesets=nodesets, facesets=facesets)
end

# TODO think about making Mesh = Grid+Topology+CoordinateSystem
generate_mesh(args...) = to_mesh(generate_grid(args...))
