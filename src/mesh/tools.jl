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

function refine_element_uniform(mgrid::SimpleMesh3D, cell::Hexahedron, cell_idx::Int, global_edge_indices, global_face_indices)
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
hexahedralize_cell(mgrid::SimpleMesh3D, cell::Hexahedron, cell_idx::Int, global_edge_indices, global_face_indices) = refine_element_uniform(mgrid, cell, cell_idx, global_edge_indices, global_face_indices)

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
    mgrid = to_mesh(grid) # Helper

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
        append!(new_cells, refine_element_uniform(mgrid, cell, cellidx, global_edge_indices, global_face_indices))
    end
    # TODO boundary sets
    return Grid(new_cells, [grid.nodes; new_edge_nodes; new_face_nodes; new_cell_nodes])
end

function hexahedralize(grid::Grid{3,C,T}) where {C,T}
    mgrid = to_mesh(grid) # Helper

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

function load_voom2_elements(filename)
    elements = Vector{Ferrite.AbstractCell}()
    open(filename, "r") do file
        # First line has format number of elements as Int and 2 more integers
        line = strip(readline(file))
        ne = parse(Int64,split(line)[1])
        resize!(elements, ne)

        while !eof(file)
            line = parse.(Int64,split(strip(readline(file))))
            ei = line[1]
            nnodes = line[2]
            if nnodes == 8
                elements[ei] = Hexahedron(ntuple(i->line[i+2],8))
            elseif nnodes == 2
                elements[ei] = Line(ntuple(i->line[i+2],2))
            else
                @error "Unknown element type $nnodes"
            end
        end
    end
    return elements
end

function load_voom2_nodes(filename)
    nodes = Vector{Ferrite.Node{3,Float64}}()
    open(filename, "r") do file
        # First line has format number of nodes as Int and 2 more integers
        line = strip(readline(file))
        nn = parse(Int64,split(line)[1])
        resize!(nodes, nn)

        while !eof(file)
            line = split(strip(readline(file)))
            ni = parse(Int64, line[1])
            nodes[ni] = Node(Vec(ntuple(i->parse(Float64,line[i+1]),3)))
        end
    end
    return nodes
end

function load_voom2_fsn(filename)
    # Big table
    f = Vector{Ferrite.Vec{3,Float64}}()
    s = Vector{Ferrite.Vec{3,Float64}}()
    n = Vector{Ferrite.Vec{3,Float64}}()
    open(filename, "r") do file
        while !eof(file)
            line = parse.(Float64,split(strip(readline(file))))
            push!(f, Vec((line[1], line[2], line[3])))
            push!(s, Vec((line[4], line[5], line[6])))
            push!(n, Vec((line[7], line[8], line[9])))
        end
    end
    return f,s,n
end
