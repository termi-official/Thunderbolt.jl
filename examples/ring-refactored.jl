include("common-stuff.jl")
using FerriteGmsh

import Ferrite: get_grid, find_field

mutable struct LoadDrivenQuasiStaticProblem{DH,CH,CV,FV,MAT,MICRO,CAL,FACE}
    # Where to put this?
    dh::DH
    ch::CH
    cv::CV
    fv::FV
    #
    material_model::MAT
    microstructure_model::MICRO
    calcium_field::CAL
    face_models::FACE
end

struct SimpleChamberContractionModel{MM, CF, FM, MM2}
    mechanical_model::MM
    calcium_field::CF
    face_models::FM
    microstructure_model::MM2
end

function Thunderbolt.semidiscretize(model::MODEL, discretization::FiniteElementDiscretization, grid::Thunderbolt.AbstractGrid) where {MODEL <: SimpleChamberContractionModel{<:QuasiStaticModel}}
    ets = elementtypes(grid)
    @assert length(ets) == 1

    ip = getinterpolation(discretization.interpolations[:displacement], getcells(grid, 1))
    ip_geo = Ferrite.default_geometric_interpolation(ip) # TODO get interpolation from cell
    dh = DofHandler(grid)
    push!(dh, :displacement, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        add!(ch, dbc)
    end
    close!(ch)

    # TODO how to deal with this?
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{ref_shape}(intorder)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    cv = CellValues(qr, ip, ip_geo)
    fv = FaceValues(qr_face, ip, ip_geo)

    #
    semidiscrete_problem = LoadDrivenQuasiStaticProblem(
        dh,
        ch,
        cv,
        fv,
        model.mechanical_model,
        model.microstructure_model,
        model.calcium_field,
        model.face_models
    )

    return semidiscrete_problem
end

# We simply unpack and assemble here
function Thunderbolt.update_linearization!(J, residual, u, problem::ProblemType, t) where { ProblemType <: LoadDrivenQuasiStaticProblem}
    @unpack dh, cv, fv, material_model, microstructure_model, calcium_field, face_models = problem

    n = ndofs_per_cell(dh)
    Jₑ = zeros(n, n)
    residualₑ = zeros(n)

    # start_assemble resets J and f
    assembler = start_assemble(J, residual)

    microstructure_cache = Thunderbolt.setup_microstructure_cache(cv, microstructure_model)
    # TODO this should be outside of this routine, because the contraction might be stateful! But where/how? Maybe all caches should be factored out...?
    contraction_cache = Thunderbolt.setup_contraction_model_cache(cv, material_model.contraction_model, calcium_field)
    element_cache = Thunderbolt.CardiacMechanicalElementCache(material_model, microstructure_cache, nothing, contraction_cache, cv)

    face_caches = ntuple(i->Thunderbolt.setup_face_cache(face_models[i], fv, t), length(face_models))

    # Loop over all cells in the grid
    # @timeit "assemble" for cell in CellIterator(dh)
    # TODO for subdofhandler in dh.subdofhandlers
    displacement_dofrange = Ferrite.dof_range(dh, :displacement)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        displacement_dofs = global_dofs[displacement_dofrange]

        # TODO refactor
        uₑ = @view u[displacement_dofs] # element dofs

        # Reinitialize cell values, and reset output arrays
        reinit!(cv, cell)
        fill!(Jₑ, 0.0)
        fill!(residualₑ, 0.0)

        # Update remaining caches specific to the element and models
        Thunderbolt.update_element_cache!(element_cache, cell, t)

        # Assemble matrix and residuals
        Thunderbolt.assemble_element!(Jₑ, residualₑ, uₑ, element_cache, t)

        for local_face_index ∈ 1:nfaces(cell)
            face_is_initialized = false
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, Thunderbolt.getboundaryname(face_cache))
                    if !face_is_initialized
                        face_is_initialized = true
                        reinit!(fv, cell, local_face_index)
                    end
                    Thunderbolt.update_face_cache(cell, face_cache, t)

                    Thunderbolt.assemble_face!(Jₑ, residualₑ, uₑ, face_cache, t)
                end
            end
        end

        assemble!(assembler, global_dofs, residualₑ, Jₑ)
    end
end

function Thunderbolt.update_linearization!(J, residual, u, problem::ProblemType, t) where { ProblemType <: LoadDrivenQuasiStaticProblem}
    @unpack dh, cv, fv, material_model, microstructure_model, calcium_field, face_models = problem

    n = ndofs_per_cell(dh)
    Jₑ = zeros(n, n)
    residualₑ = zeros(n)

    # start_assemble resets J and f
    assembler = start_assemble(J, residual)

    microstructure_cache = Thunderbolt.setup_microstructure_cache(cv, microstructure_model)
    # TODO this should be outside of this routine, because the contraction might be stateful! But where/how? Maybe all caches should be factored out...?
    contraction_cache = Thunderbolt.setup_contraction_model_cache(cv, material_model.contraction_model, calcium_field)
    element_cache = Thunderbolt.CardiacMechanicalElementCache(material_model, microstructure_cache, nothing, contraction_cache, cv)

    face_caches = ntuple(i->Thunderbolt.setup_face_cache(face_models[i], fv, t), length(face_models))

    # Loop over all cells in the grid
    # @timeit "assemble" for cell in CellIterator(dh)
    # TODO for subdofhandler in dh.subdofhandlers
    displacement_dofrange = Ferrite.dof_range(dh, :displacement)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        displacement_dofs = global_dofs[displacement_dofrange]

        # TODO refactor
        uₑ = @view u[displacement_dofs] # element dofs

        # Reinitialize cell values, and reset output arrays
        reinit!(cv, cell)
        fill!(Jₑ, 0.0)
        fill!(residualₑ, 0.0)

        # Update remaining caches specific to the element and models
        Thunderbolt.update_element_cache!(element_cache, cell, t)

        # Assemble matrix and residuals
        Thunderbolt.assemble_element!(Jₑ, residualₑ, uₑ, element_cache, t)

        for local_face_index ∈ 1:nfaces(cell)
            face_is_initialized = false
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, Thunderbolt.getboundaryname(face_cache))
                    if !face_is_initialized
                        face_is_initialized = true
                        reinit!(fv, cell, local_face_index)
                    end
                    Thunderbolt.update_face_cache(cell, face_cache, t)

                    Thunderbolt.assemble_face!(Jₑ, residualₑ, uₑ, face_cache, t)
                end
            end
        end

        assemble!(assembler, global_dofs, residualₑ, Jₑ)
    end
end

# mutable struct AverageCoordinateStrainProcessorCache{CC, SVT <: AbstractVector}
#     coordinate_system_cache::CC
#     E_cc::SVT
#     E_ll::SVT
#     E_rr::SVT
# end

# function prepare_postprocessor_cache!(cache::AverageCoordinateStrainProcessorCache{CC, SVT}, problem) where {CC, SVT <: AbstractVector}
#     @unpack dh   = problem
#     @unpack grid = dh
#     _ncells = getncells(grid)
#     cache.E_cc = zero(SVT(undef, _ncells))
#     cache.E_ll = zero(SVT(undef, _ncells))
#     cache.E_rr = zero(SVT(undef, _ncells))
# end

# mutable struct AverageCardiacMechanicalMicrostructuralPostProcessorCache{MC, SVT <: AbstractVector, VVT <: AbstractVector}
#     microstructure_cache::MC
#     E_ff:SVT
#     frefdata::VVT
#     fdata::VVT
#     srefdata::VVT
#     sdata::VVT
#     nrefdata::VVT
#     ndata::VVT
#     helixangledata::SVT
#     helixanglerefdata::SVT
# end

# function prepare_postprocessor_cache!(cache::AverageCardiacMechanicalMicrostructuralPostProcessorCache{MC, SVT, VVT}, problem, t, uₜ) where {MC, SVT <: AbstractVector, VVT <: AbstractVector}
#     @unpack dh   = problem
#     @unpack grid = dh
#     @assert getdim(grid) == 3
#     _ncells = getncells(grid)
#     cache.E_ff = zero(SVT(undef,_ncells))
#     cache.frefdata = zero(VVT(undef, _ncells))
#     cache.srefdata = zero(VVT(undef, _ncells))
#     cache.nrefdata = zero(VVT(undef, _ncells))
#     cache.fdata = zero(VVT(undef, _ncells))
#     cache.sdata = zero(VVT(undef, _ncells))
#     cache.ndata = zero(VVT(undef, _ncells))
#     helixangledata = zero(SVT(undef,_ncells))
#     helixanglerefdata = zero(SVT(undef,_ncells))
# end

# mutable struct VolumePostProcessorCache{SVT <: AbstractVector}
#     detF_per_cell::SVT
# end

# function prepare_postprocessor_cache!(cache::VolumePostProcessorCache{SVT}, problem, t, uₜ) where {SVT <: AbstractVector}
#     @unpack dh = problem
#     grid = getgrid(dh)
#     _ncells = getncells(grid)
#     cache.detF_per_cell = zero(SVT(undef, _ncells))
# end

# struct StandardIOPostProcessor{IO, CV, CL}
#     io::IO
#     cv::CV
#     subdomains::Vector{Int}
#     # cache_list::CL
# end

struct StandardMechanicalIOPostProcessor{IO, CV, MC}
    io::IO
    cv::CV
    subdomains::Vector{Int}
    # coordinate_system_cache::CC
    microstructure_cache::MC
end

function (postproc::StandardMechanicalIOPostProcessor{IO, CV, MC})(t, problem, solver_cache) where {IO, CV, MC}
    @unpack dh = problem
    grid = get_grid(dh)
    @unpack io, cv, subdomains, microstructure_cache = postproc

    # Compute some elementwise measures
    E_ff = zeros(getncells(grid))
    E_ff2 = zeros(getncells(grid))
    E_cc = zeros(getncells(grid))
    E_ll = zeros(getncells(grid))
    E_rr = zeros(getncells(grid))

    Jdata = zeros(getncells(grid))

    frefdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    srefdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    fdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    sdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    helixangledata = zero(Vector{Float64}(undef, getncells(grid)))
    helixanglerefdata = zero(Vector{Float64}(undef, getncells(grid)))

    # Compute some elementwise measures
    for sdh ∈ dh.subdofhandlers[postproc.subdomains]
        field_idx = find_field(sdh, :displacement)
        field_idx === nothing && continue 
        for cell ∈ CellIterator(sdh)
            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            field_dofs  = dof_range(sdh, field_idx)
            uₑ = solver_cache.uₜ[global_dofs] # element dofs

            update_microstructure_cache!(microstructure_cache, t, cell, cv)

            E_ff_cell = 0.0
            E_cc_cell = 0.0
            E_rr_cell = 0.0
            E_ll_cell = 0.0

            Jdata_cell = 0.0

            frefdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            srefdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            fdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            sdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            helixangle_cell = 0.0
            helixangleref_cell = 0.0

            nqp = getnquadpoints(cv)
            for qp in 1:nqp
                dΩ = getdetJdV(cv, qp)

                # Compute deformation gradient F
                ∇u = function_gradient(cv, qp, uₑ)
                F = one(∇u) + ∇u

                C = tdot(F)
                E = (C-one(C))/2.0
                f₀,s₀,n₀ = directions(microstructure_cache, qp)

                E_ff_cell += f₀ ⋅ E ⋅ f₀

                f₀_current = F⋅f₀
                f₀_current /= norm(f₀_current)

                s₀_current = F⋅s₀
                s₀_current /= norm(s₀_current)

                coords = getcoordinates(cell)
                x_global = spatial_coordinate(cv, qp, coords)

                # v_longitudinal = function_gradient(cv_cs, qp, coordinate_system.u_apicobasal[celldofs(cell)])
                # v_radial = function_gradient(cv_cs, qp, coordinate_system.u_transmural[celldofs(cell)])
                # v_circimferential = v_longitudinal × v_radial
                # @TODO compute properly via coordinate system
                v_longitudinal = Ferrite.Vec{3}((0.0, 0.0, 1.0))
                v_radial = Ferrite.Vec{3}((x_global[1],x_global[2],0.0))
                v_radial /= norm(v_radial)
                v_circimferential = v_longitudinal × v_radial # Ferrite.Vec{3}((x_global[2],x_global[1],0.0))
                v_circimferential /= norm(v_circimferential)
                #
                E_ll_cell += v_longitudinal ⋅ E ⋅ v_longitudinal
                E_rr_cell += v_radial ⋅ E ⋅ v_radial
                E_cc_cell += v_circimferential ⋅ E ⋅ v_circimferential

                Jdata_cell += det(F)

                frefdata_cell += f₀
                srefdata_cell += s₀

                fdata_cell += f₀_current
                sdata_cell += s₀_current

                helixangle_cell += acos(clamp(f₀_current ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀_current) ⋅ v_radial)
                helixangleref_cell += acos(clamp(f₀ ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀) ⋅ v_radial)
            end

            E_ff[Ferrite.cellid(cell)] = E_ff_cell / nqp
            E_cc[Ferrite.cellid(cell)] = E_cc_cell / nqp
            E_rr[Ferrite.cellid(cell)] = E_rr_cell / nqp
            E_ll[Ferrite.cellid(cell)] = E_ll_cell / nqp
            Jdata[Ferrite.cellid(cell)] = Jdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] = frefdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] /= norm(frefdata[Ferrite.cellid(cell)])
            srefdata[Ferrite.cellid(cell)] = srefdata_cell / nqp
            srefdata[Ferrite.cellid(cell)] /= norm(srefdata[Ferrite.cellid(cell)])
            fdata[Ferrite.cellid(cell)] = fdata_cell / nqp
            fdata[Ferrite.cellid(cell)] /= norm(fdata[Ferrite.cellid(cell)])
            sdata[Ferrite.cellid(cell)] = sdata_cell / nqp
            sdata[Ferrite.cellid(cell)] /= norm(sdata[Ferrite.cellid(cell)])
            helixanglerefdata[Ferrite.cellid(cell)] = helixangleref_cell / nqp
            helixangledata[Ferrite.cellid(cell)] = helixangle_cell / nqp
        end
    end

    # Save the solution
    Thunderbolt.store_timestep!(io, t, dh.grid)
    Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₜ, :displacement)
    Thunderbolt.store_timestep_celldata!(io, t, hcat(frefdata...),"Reference Fiber Data")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(fdata...),"Current Fiber Data")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(srefdata...),"Reference Sheet Data")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(sdata...),"Current Sheet Data")
    Thunderbolt.store_timestep_celldata!(io, t, E_ff,"E_ff")
    Thunderbolt.store_timestep_celldata!(io, t, E_ff2,"E_ff2")
    Thunderbolt.store_timestep_celldata!(io, t, E_cc,"E_cc")
    Thunderbolt.store_timestep_celldata!(io, t, E_rr,"E_rr")
    Thunderbolt.store_timestep_celldata!(io, t, E_ll,"E_ll")
    Thunderbolt.store_timestep_celldata!(io, t, Jdata,"J")
    Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixangledata),"Helix Angle")
    Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixanglerefdata),"Helix Angle (End Diastole)")
    Thunderbolt.finalize_timestep!(io, t)

    # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
    # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
end

# function solve_test_ring(name_base, material_model, grid, coordinate_system, microstructure_model, face_models, calcium_field, ip_mech, ip_geo, intorder::Int, Δt = 0.1, T = 2.0)
function solve_test_ring(name_base, material_model, grid, microstructure_model, face_models::FM, calcium_field, ip_mech::IPM, ip_geo::IPG, intorder::Int, Δt = 0.1, T = 2.0) where {ref_shape, IPM <: Interpolation{ref_shape}, IPG <: Interpolation{ref_shape}, FM}
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    problem = semidiscretize(
        SimpleChamberContractionModel(material_model, calcium_field, face_models, microstructure_model),
        FiniteElementDiscretization(
            Dict(:displacement => LagrangeCollection{1}()^3),
            [Dirichlet(:displacement, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])],
            # [Dirichlet(:displacement, getfaceset(grid, "left"), (x,t) -> [0.0], [1]),Dirichlet(:displacement, getfaceset(grid, "front"), (x,t) -> [0.0], [2]),Dirichlet(:displacement, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3]), Dirichlet(:displacement, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])],
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValues(QuadratureRule{ref_shape}(intorder-1), ip_mech, ip_geo)
    microstructure_cache = setup_microstructure_cache(cv_post, microstructure_model)
    standard_postproc = StandardMechanicalIOPostProcessor(io, cv_post, [1], microstructure_cache)

    # Create sparse matrix and residual vector
    solver = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100))

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        nothing,
        standard_postproc
    )
end

import Thunderbolt: QuasiStaticModel
"""
    QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A discrete problem with time dependent terms and no time derivatives w.r.t. any solution variable.
Abstractly written we want to solve the problem F(u, t) = 0 on some time interval [t₁, t₂].
"""
struct QuasiStaticNonlinearProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}
    model::M
    dh::DH
end

"""
    QuasiStaticDAEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}

A problem with time dependent terms and time derivatives only w.r.t. internal solution variable.
"""
struct QuasiStaticDAEProblem{M <: QuasiStaticModel, DH <: Ferrite.AbstractDofHandler}
    model::M
    dh::DH
end

# TODO better model struct
# function Thunderbolt.semidiscretize(model::Tuple{QuasiStaticModel,CM,FM,MM}, discretization::FiniteElementDiscretization, grid::Thunderbolt.AbstractGrid) where {CM, FM, MM}
#     ets = elementtypes(grid)
#     @assert length(ets) == 1

#     ip = getinterpolation(discretization.interpolations[:displacement], getcells(grid, 1))
#     ip_geo = Ferrite.default_geometric_interpolation(ip) # TODO get interpolation from cell
#     dh = DofHandler(grid)
#     push!(dh, :displacement, ip)
#     close!(dh);

#     ch = ConstraintHandler(dh)
#     for dbc ∈ discretization.dbcs
#         add!(ch, dbc)
#     end
#     close!(ch)

#     # TODO how to deal with this?
#     intorder = 2*Ferrite.getorder(ip)
#     ref_shape = Ferrite.getrefshape(ip)
#     qr = QuadratureRule{ref_shape}(intorder)
#     qr_face = FaceQuadratureRule{ref_shape}(intorder)
#     cv = CellValues(qr, ip, ip_geo)
#     fv = FaceValues(qr_face, ip, ip_geo)

#     #
#     semidiscrete_problem = LoadDrivenQuasiStaticProblem(
#         dh,
#         ch,
#         cv,
#         fv,
#         model[1],
#         model[4],
#         model[2],
#         model[3],
#         # Belongs to the solver.
#         #zeros(ndofs(dh)),
#         #0.0,
#         #0.0,
#     )

#     return semidiscrete_problem
# end

# for (filename, ref_shape, order) ∈ [
#     # ("MidVentricularSectionQuadTet.msh", RefTetrahedron, 2),
#     ("MidVentricularSectionTet.msh", RefTetrahedron, 1),
#     # ("MidVentricularSectionHex.msh", RefCube, 1),
#     # ("MidVentricularSectionQuadHex.msh", RefCube, 2) # We have to update FerriteGmsh first, because the hex27 translator is missing. See https://github.com/Ferrite-FEM/FerriteGmsh.jl/pull/29
# ]
function run_simulations()

ref_shape = RefHexahedron
order = 1

ip_fiber = Lagrange{ref_shape, order}()
ip_u = Lagrange{ref_shape, order}()^3
ip_geo = Lagrange{ref_shape, order}()

ring_grid = generate_ring_mesh(8,2,2)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
solve_test_ring("Debug",
    ActiveStressModel(
        Guccione1991Passive(),
        Guccione1993Active(10.0),
        PelceSunLangeveld1995Model()
    ), ring_grid, 
    create_simple_fiber_model(ring_cs, ip_fiber, ip_geo,
        endo_helix_angle = deg2rad(0.0),
        epi_helix_angle = deg2rad(0.0),
        endo_transversal_angle = 0.0,
        epi_transversal_angle = 0.0,
        sheetlet_pseudo_angle = deg2rad(0)
    ),
    [NormalSpringBC(0.01, "Epicardium")],
    CalciumHatField(), ip_u, ip_geo, 2*order,
    0.1
)

return

ring_grid = generate_ring_mesh(50,10,10)
filename = "MidVentricularSectionHexG50-10-10"

# ring_grid = saved_file_to_grid("../data/meshes/ring/" * filename)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
ring_fm = create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(60.0), epi_helix_angle = deg2rad(-60.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0)

passive_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))

# solve_test_ring(filename*"_GHM-HO_AS1_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ),
#     ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# Diverges...?
# solve_test_ring(filename*"_GHM-HO_AS2_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring2()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_HO_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(passive_model),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Simple_Pelce",
#     ActiveStressModel(
#         passive_model,
#         SimpleActiveStress(),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Piersanti_Pelce",
#     ActiveStressModel(
#         passive_model,
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HONEW_HONEW_RLRSQ100_Pelce",
#     ExtendedHillModel(
#         NewHolzapfelOgden2009Model(),
#         ActiveMaterialAdapter(NewHolzapfelOgden2009Model(;mpU=NullCompressionPenalty())),
#         RLRSQActiveDeformationGradientModel(0.5),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HONEW_Piersanti_Pelce",
#     ActiveStressModel(
#         NewHolzapfelOgden2009Model(),
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"Vallespin2023-Reproducer",
#     ActiveStressModel(
#         Guccione1991Passive(),
#         Guccione1993Active(150.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.01, "Epicardium")],
#     CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"Vallespin2023-Ring",
#     ActiveStressModel(
#         Guccione1991Passive(),
#         Guccione1993Active(10),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs,
#     create_simple_fiber_model(ring_cs, ip_fiber, ip_geo,
#         endo_helix_angle = deg2rad(60.0),
#         epi_helix_angle = deg2rad(-60.0),
#         endo_transversal_angle = 0.0,
#         epi_transversal_angle = 0.0,
#         sheetlet_pseudo_angle = deg2rad(20)
#     ),
#     [NormalSpringBC(0.01, "Epicardium")],
#     CalciumHatField(), ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce_MoulinHelixAngle",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(50.0), epi_helix_angle = deg2rad(-40.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce_MoulinHelixAngle_SA45",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_helix_angle = deg2rad(50.0), epi_helix_angle = deg2rad(-40.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_pseudo_angle = deg2rad(45)),
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_u, ip_geo, 2*order
# )

end

run_simulations()
