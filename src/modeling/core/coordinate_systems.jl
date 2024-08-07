"""
    CartesianCoordinateSystem(mesh)

Standard cartesian coordinate system.
"""
struct CartesianCoordinateSystem{sdim}
end

value_type(::CartesianCoordinateSystem{sdim}) where sdim = Vec{sdim}

CartesianCoordinateSystem(mesh::AbstractGrid{sdim}) where sdim = CartesianCoordinateSystem{sdim}()

"""
    getcoordinateinterpolation(cs::CartesianCoordinateSystem, cell::AbstractCell)

Get interpolation function for the cartesian coordinate system.
"""
getcoordinateinterpolation(cs::CartesianCoordinateSystem{sdim}, cell::CellType) where {sdim, CellType <: AbstractCell} = Ferrite.geometric_interpolation(CellType)^sdim


"""
    LVCoordinateSystem(dh, u_transmural, u_apicobasal)

Simplified universal ventricular coordinate on LV only, containing the transmural, apicobasal and
circumferential coordinates. See [`compute_lv_coordinate_system`](@ref) to construct it.
"""
struct LVCoordinateSystem{DH <: AbstractDofHandler, IPC}
    dh::DH
    ip_collection::IPC # TODO special dof handler with type stable interpolation
    u_transmural::Vector{Float64}
    u_apicobasal::Vector{Float64}
    u_circumferential::Vector{Float64}
end


"""
    LVCoordinate{T}

LV only part of the universal ventricular coordinate, containing
    * transmural
    * apicobasal
    * circumferential
"""
struct LVCoordinate{T}
    transmural::T
    apicaobasal::T
    circumferential::T
end

value_type(::LVCoordinateSystem) = LVCoordinate


"""
    getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell)

Get interpolation function for the LV coordinate system.
"""
getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell) = getinterpolation(cs.ip_collection, cell)


"""
    compute_lv_coordinate_system(mesh::SimpleMesh)

Requires a mesh with facetsets
    * Base
    * Epicardium
    * Endocardium
and a nodeset
    * Apex

!!! warning
    The circumferential coordinate is not yet implemented and is guaranteed to evaluate to NaN.
"""
function compute_lv_coordinate_system(mesh::SimpleMesh{3}, subdomains::Vector{String} = [""])
    ip_collection = LagrangeCollection{1}()
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection)
    cellvalues = getcellvalues(cv_collection, getcells(mesh, 1))

    dh = DofHandler(mesh)
    for name in subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(:coordinates, ip_collection)])
    end
    Ferrite.close!(dh)

    # Assemble Laplacian
    # TODO use bilinear operator for performance
    K = create_sparsity_pattern(dh)

    assembler = start_assemble(K)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(mesh, first(sdh.cellset)))
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        @inbounds for cell in CellIterator(sdh)
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
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Endocardium"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Epicardium"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    # Apicobasal coordinate
    #TODO refactor check for node set existence
    if !haskey(mesh.grid.nodesets, "Apex") #TODO this is just a hotfix, assuming that z points towards the apex
        apex_node_index = 1
        nodes = getnodes(mesh)
        for (i,node) ∈ enumerate(nodes)
            if nodes[i].x[3] > nodes[apex_node_index].x[3]
                apex_node_index = i
            end
        end
        addnodeset!(mesh, "Apex", OrderedSet{Int}((apex_node_index)))
    end

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Base"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getnodeset(mesh, "Apex"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_apicobasal = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_apicobasal, f, ch)
    apicobasal = K_apicobasal \ f;

    circumferential = zeros(ndofs(dh))
    circumferential .= NaN

    return LVCoordinateSystem(dh, ip_collection, transmural, apicobasal, circumferential)
end

"""
    compute_midmyocardial_section_coordinate_system(mesh::SimpleMesh)

Requires a mesh with facetsets
    * Base
    * Epicardium
    * Endocardium
    * Myocardium
"""
function compute_midmyocardial_section_coordinate_system(mesh::SimpleMesh{3}, subdomains::Vector{String} = [""])
    ip_collection = LagrangeCollection{1}()
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection)

    dh = DofHandler(mesh)
    for name in subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(:coordinates, ip_collection)])
    end
    Ferrite.close!(dh)

    # Assemble Laplacian
    # TODO use bilinear operator
    K = create_sparsity_pattern(dh)

    assembler = start_assemble(K)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(mesh, first(sdh.cellset)))
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        @inbounds for cell in CellIterator(sdh)
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
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Endocardium"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Epicardium"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Base"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Myocardium"), (x, t) -> 0.15)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_apicobasal = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_apicobasal, f, ch)
    apicobasal = K_apicobasal \ f;

    circumferential = zeros(ndofs(dh))
    circumferential .= NaN

    return LVCoordinateSystem(dh, ip_collection, transmural, apicobasal, circumferential)
end

"""
    vtk_coordinate_system(vtk, cs::LVCoordinateSystem)

Store the LV coordinate system in a vtk file.
"""
function vtk_coordinate_system(vtk, cs::LVCoordinateSystem)
    vtk_point_data(vtk, cs.dh, cs.u_apicobasal, "apicobasal_")
    vtk_point_data(vtk, cs.dh, cs.u_transmural, "transmural_")
end

"""
    BiVCoordinateSystem(dh, u_transmural, u_apicobasal, u_rotational, u_transventricular)

Universal ventricular coordinate, containing the transmural, apicobasal, circumferential 
and transventricular coordinates.
"""
struct BiVCoordinateSystem{DH <: Ferrite.AbstractDofHandler}
    dh::DH
    u_transmural::Vector{Float32}
    u_apicobasal::Vector{Float32}
    u_rotational::Vector{Float32}
    u_transventricular::Vector{Float32}
end


"""
BiVCoordinate{T}

Biventricular universal coordinate, containing
    * transmural
    * apicobasal
    * rotational
    * transventricular
"""
struct BiVCoordinate{T}
    transmural::T
    apicaobasal::T
    rotational::T
    transventricular::T
end

value_type(::BiVCoordinateSystem) = BiVCoordinate

getcoordinateinterpolation(cs::BiVCoordinateSystem, cell::Ferrite.AbstractCell) = Ferrite.getfieldinterpolation(cs.dh, (1,1))

function vtk_coordinate_system(vtk, cs::BiVCoordinateSystem)
    vtk_point_data(vtk, bivcs.dh, bivcs.u_transmural, "_transmural")
    vtk_point_data(vtk, bivcs.dh, bivcs.u_apicobasal, "_apicobasal")
    vtk_point_data(vtk, bivcs.dh, bivcs.u_rotational, "_rotational")
    vtk_point_data(vtk, bivcs.dh, bivcs.u_transventricular, "_transventricular")
end

