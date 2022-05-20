# Generates a hexahedral ring in the form of a large cylinder subtracted by a smaller cylinder.
function generate_ring_mesh(ne_c, ne_r, ne_l; radial_inner::T = Float64(0.75), radial_outer::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = ne_c*ne_r*ne_l;
    n_nodes_c = ne_c; n_nodes_r = ne_r+1; n_nodes_l = ne_l+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    coords_c = range(0.0, stop=2*π, length=n_nodes_c+1)
    coords_r = range(radial_inner, stop=radial_outer, length=n_nodes_r)
    coords_l = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = coords_r[j]-0.03*coords_l[k]/maximum(abs.(coords_l))
        push!(nodes, Node((radius*cos(coords_c[i]), radius*sin(coords_c[i]), coords_l[k])))
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

# Generates a hexahedral truncated ellipsoidal mesh.
function generate_ideal_lv_mesh()

end