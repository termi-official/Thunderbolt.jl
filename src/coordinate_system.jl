"""
Simple Coordinate System.
"""
struct LVCoordinateSystem
    dh::AbstractDofHandler
    u_transmural::Vector{Float64}
    u_apicobasal::Vector{Float64}
end

"""
"""
getcoordinateinterpolation(cs::LVCoordinateSystem) = getfieldinterpolation(dh, 1)

"""
Requires a grid with facesets
* Base
* Epicardium
* Endocardium
and a nodeset
* Apex
"""
function compute_LV_coordinate_system(grid::AbstractGrid,ip_geo;ref_shape=RefTetrahedron)
    ip = Lagrange{3, ref_shape, 1}()
    qr = QuadratureRule{3, ref_shape}(2)
    cellvalues = CellScalarValues(qr, ip, ip_geo);

    dh = DofHandler(grid)
    push!(dh, :coordinates, 1, ip)
    Ferrite.close!(dh);

    # Assemble Laplacian
    K = create_sparsity_pattern(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(K)
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Ke)
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Endocardium"), (x, t) -> 0)
    add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Epicardium"), (x, t) -> 1)
    add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    # Apicobasal coordinate
    #TODO refactor check for node set existence
    if !haskey(getnodesets(grid), "Apex") #TODO this is just a hotfix, assuming that z points towards the apex
        apex_node_index = 1
        nodes = getnodes(grid)
        for (i,node) ∈ enumerate(nodes)
            if nodes[i].x[3] > nodes[apex_node_index].x[3]
                apex_node_index = i
            end
        end
        addnodeset!(grid, "Apex", Set{Int}((apex_node_index)))
    end

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Base"), (x, t) -> 0)
    add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getnodeset(grid, "Apex"), (x, t) -> 1)
    add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_apicobasal = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_apicobasal, f, ch)
    apicobasal = K_apicobasal \ f;

    return LVCoordinateSystem(dh, transmural, apicobasal)
end

"""
"""
function compute_midmyocardial_section_coordinate_system(grid::AbstractGrid,ip_geo::Interpolation;ref_shape=RefTetrahedron)
    ip = Lagrange{3, ref_shape, 1}()
    qr = QuadratureRule{3, ref_shape}(2)
    cellvalues = CellScalarValues(qr, ip, ip_geo);

    dh = DofHandler(grid)
    push!(dh, :coordinates, 1, ip)
    Ferrite.close!(dh);

    # Assemble Laplacian
    K = create_sparsity_pattern(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(K)
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Ke)
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Endocardium"), (x, t) -> 0)
    add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Epicardium"), (x, t) -> 1)
    add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Base"), (x, t) -> 0)
    add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Myocardium"), (x, t) -> 0.15)
    add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_apicobasal = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_apicobasal, f, ch)
    apicobasal = K_apicobasal \ f;

    return LVCoordinateSystem(dh, transmural, apicobasal)
end

"""
"""
function vtk_coordinate_system(vtk, cs::LVCoordinateSystem)
    vtk_point_data(vtk, cs.dh, cs.u_apicobasal, "apicobasal_")
    vtk_point_data(vtk, cs.dh, cs.u_transmural, "transmural_")
end
