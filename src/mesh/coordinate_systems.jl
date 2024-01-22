
struct CartesianCoordinateSystem end

"""
Simple Coordinate System.

!!! note TODO implement circumferential coordinate
"""
struct LVCoordinateSystem
    dh::AbstractDofHandler
    u_transmural::Vector{Float64}
    u_apicobasal::Vector{Float64}
end

"""
"""
getcoordinateinterpolation(cs::LVCoordinateSystem) = Ferrite.getfieldinterpolation(cs.dh, (1,1))

"""
"""
create_cellvalues(cs::LVCoordinateSystem, qr::QuadratureRule, ip_geo=getcoordinateinterpolation(cs)) = CellValues(qr, getcoordinateinterpolation(cs), ip_geo)

"""
Requires a grid with facesets
* Base
* Epicardium
* Endocardium
and a nodeset
* Apex
"""
function compute_LV_coordinate_system(grid::AbstractGrid{dim}, ip_geo_collection::LagrangeCollection) where {dim}
    @assert dim == 3
    @assert length(elementtypes(grid)) == 1
    ref_shape = getrefshape(getcells(grid, 1))
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, ref_shape)
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection, ip_geo_collection)
    cellvalues = getcellvalues(cv_collection, getcells(grid, 1))

    dh = DofHandler(grid)
    Ferrite.add!(dh, :coordinates, ip)
    Ferrite.close!(dh)

    # Assemble Laplacian
    K = create_sparsity_pattern(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(K)
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)

        for qp in QuadratureIterator(cellvalues)
            dΩ = getdetJdV(cellvalues, qp)

            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, qp, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, qp, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Ke)
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Endocardium"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Epicardium"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    # Apicobasal coordinate
    #TODO refactor check for node set existence
    if !haskey(grid.nodesets, "Apex") #TODO this is just a hotfix, assuming that z points towards the apex
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
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getnodeset(grid, "Apex"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
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
function compute_midmyocardial_section_coordinate_system(grid::AbstractGrid{dim}, ip_geo_collection::InterpolationCollection) where {dim}
    @assert dim == 3
    @assert length(elementtypes(grid)) == 1
    ref_shape = getrefshape(getcells(grid,1))
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, ref_shape)
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection, ip_geo_collection)
    cellvalues = getcellvalues(cv_collection, getcells(grid,1))

    dh = DofHandler(grid)
    Ferrite.add!(dh, :coordinates, ip)
    Ferrite.close!(dh);

    # Assemble Laplacian
    K = create_sparsity_pattern(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(K)
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)

        reinit!(cellvalues, cell)

        for qp in QuadratureIterator(cellvalues)
            dΩ = getdetJdV(cellvalues, qp)

            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, qp, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, qp, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Ke)
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Endocardium"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Epicardium"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Base"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfaceset(grid, "Myocardium"), (x, t) -> 0.15)
    Ferrite.add!(ch, dbc);
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
