"""
    CartesianCoordinateSystem(mesh)

Standard cartesian coordinate system.
"""
struct CartesianCoordinateSystem{sdim}
end

CartesianCoordinateSystem(grid::AbstractGrid{sdim}) where sdim = CartesianCoordinateSystem{sdim}()

"""
    getcoordinateinterpolation(cs::CartesianCoordinateSystem, cell::AbstractCell)

Get interpolation function for the cartesian coordinate system.
"""
getcoordinateinterpolation(cs::CartesianCoordinateSystem{sdim}, cell::CellType) where {sdim, CellType <: AbstractCell} = Ferrite.default_interpolation(CellType)^sdim


"""
    LVCoordinateSystem(dh, u_transmural, u_apicobasal)

Simplified universal ventricular coordinate on LV only, containing the transmural, apicobasal and
circumferential coordinates. See [`compute_LV_coordinate_system`](@ref) to construct it.
"""
struct LVCoordinateSystem{DH <: AbstractDofHandler}
    dh::DH
    u_transmural::Vector{Float64}
    u_apicobasal::Vector{Float64}
    u_circumferential::Vector{Float64}
    function LVCoordinateSystem(dh::AbstractDofHandler, u_transmural::Vector{Float64}, u_apicobasal::Vector{Float64}, u_circumferential::Vector{Float64})
        check_subdomains(dh)
        return new{typeof(dh)}(dh, u_transmural, u_apicobasal, u_circumferential)
    end
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


"""
    getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell)

Get interpolation function for the LV coordinate system.
"""
getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell) = Ferrite.getfieldinterpolation(cs.dh, (1,1))


"""
    compute_LV_coordinate_system(grid::AbstractGrid)

Requires a grid with facesets
    * Base
    * Epicardium
    * Endocardium
and a nodeset
    * Apex

!!! warning
    The circumferential coordinate is not yet implemented and is guaranteed to evaluate to NaN.
"""
function compute_LV_coordinate_system(grid::AbstractGrid{3})
    check_subdomains(grid)

    ref_shape = getrefshape(getcells(grid, 1))
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, ref_shape)
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection)
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

    circumferential = zeros(ndofs(dh))
    circumferential .= NaN

    return LVCoordinateSystem(dh, transmural, apicobasal, circumferential)
end

"""
    compute_midmyocardial_section_coordinate_system(grid::AbstractGrid)

Requires a grid with facesets
    * Base
    * Epicardium
    * Endocardium
    * Myocardium
"""
function compute_midmyocardial_section_coordinate_system(grid::AbstractGrid{dim}) where {dim}
    @assert dim == 3
    @assert length(elementtypes(grid)) == 1
    ref_shape = getrefshape(getcells(grid,1))
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, ref_shape)
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection)
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

    circumferential = zeros(ndofs(dh))
    circumferential .= NaN

    return LVCoordinateSystem(dh, transmural, apicobasal, circumferential)
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

Thunderbolt.getcoordinateinterpolation(cs::BiVCoordinateSystem, cell::Ferrite.AbstractCell) = Ferrite.getfieldinterpolation(cs.dh, (1,1))

function Thunderbolt.evaluate_coefficient(coeff::CoordinateSystemCoefficient{<:BiVCoordinateSystem}, cell_cache, qp::QuadraturePoint{ref_shape,T}, t) where {ref_shape,T}
    x = @MVector zeros(T, 4)
    ip = getcoordinateinterpolation(coeff.cs, getcells(cell_cache.grid, cellid(cell_cache)))
    dofs = celldofs(coeff.cs.dh, cellid(cell_cache))
    @inbounds for i in 1:getnbasefunctions(ip)
        val = Ferrite.shape_value(ip, qp.ξ, i)
        x[1] += val * coeff.cs.u_transmural[dofs[i]]
        x[2] += val * coeff.cs.u_apicobasal[dofs[i]]
        x[3] += val * coeff.cs.u_rotational[dofs[i]]
        x[4] += val * coeff.cs.u_transventricular[dofs[i]]
    end
    return BiVCoordinate(x[1], x[2], x[3], x[4])
end

function Thunderbolt.vtk_coordinate_system(vtk, cs::BiVCoordinateSystem)
    vtk_point_data(vtk, bivcs.dh, bivcs.u_transmural, "_transmural")
    vtk_point_data(vtk, bivcs.dh, bivcs.u_apicobasal, "_apicobasal")
    vtk_point_data(vtk, bivcs.dh, bivcs.u_rotational, "_rotational")
    vtk_point_data(vtk, bivcs.dh, bivcs.u_transventricular, "_transventricular")
end
