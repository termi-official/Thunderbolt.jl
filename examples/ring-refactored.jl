using Thunderbolt, UnPack, SparseArrays
import Thunderbolt: Ψ, U

# struct MicrostructureCache{sdim}
#     reference_fibers::Vector{Vec{sdim}}
#     reference_sheets::Vector{Vec{sdim}}
#     reference_normals::Vector{Vec{sdim}}
# end

# function directions(cache::MicrostructureCache{dim}, qp::Int) where {dim}
#     reference_fibers[qp], reference_sheets[qp], reference_normals[qp]
# end


# ----------------------------------------------

mutable struct LazyMicrostructureCache{MM, VT}
    const microstructure_model::MM
    const x_ref::Vector{VT}
    cellid::Int
end

function Thunderbolt.directions(cache::LazyMicrostructureCache{MM}, qp::Int) where {MM}
    return directions(cache.microstructure_model, cache.cellid, cache.x_ref[qp])
end

function setup_microstructure_cache(cv, model::OrthotropicMicrostructureModel{FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}) where {FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}
    return LazyMicrostructureCache(model, cv.qr.points, -1)
end

function update_microstructure_cache!(cache::LazyMicrostructureCache{MM}, time::Float64, cell::CellCacheType, cv::CV) where {CellCacheType, CV, MM}
    cache.cellid = cellid(cell)
end


# ----------------------------------------------


struct PelceSunLangeveld1995Cache
    calcium_values::Vector{Float64}
end

function state(cache::PelceSunLangeveld1995Cache, qp::Int)
    return cache.calcium_values[qp]
end

function setup_contraction_model_cache(cv::CV, contraction_model::PelceSunLangeveld1995Model, cf::CF) where {CV, CF}
    return PelceSunLangeveld1995Cache(Vector{Float64}(undef, getnquadpoints(cv)))
end

function update_contraction_model_cache!(cache::PelceSunLangeveld1995Cache, time::Float64, cell::CellCacheType, cv::CV, calcium_field::CF) where {CellCacheType, CV, CF}
    for qp ∈ 1:getnquadpoints(cv)
        x_ref = cv.qr.points[qp]
        cache.calcium_values[qp] = value(calcium_field, Ferrite.cellid(cell), x_ref, time)
    end
end


# ----------------------------------------------


struct CardiacMechanicalElementCache{MP, MSCache, CMCache, CV}
    mp::MP
    microstructure_cache::MSCache
    # coordinate_system_cache::CSCache
    contraction_model_cache::CMCache
    cv::CV
end

function update_element_cache!(cache::CardiacMechanicalElementCache{MP, MSCache, CMCache, CV}, calcium_field::CF, time::Float64, cell::CellCacheType) where {CellCacheType, MP, MSCache, CMCache, CV, CF}
    update_microstructure_cache!(cache.microstructure_cache, time, cell, cache.cv)
    update_contraction_model_cache!(cache.contraction_model_cache, time, cell, cache.cv, calcium_field)
end

function assemble_element!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, vₑ::Vector, cache::CardiacMechanicalElementCache)
    # @unpack mp, microstructure_cache, coordinate_system_cache, cv, contraction_model_cache = cache
    @unpack mp, microstructure_cache, contraction_model_cache, cv = cache
    ndofs = getnbasefunctions(cv)

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        f₀, s₀, n₀ = directions(microstructure_cache, qp)
        contraction_state = state(contraction_model_cache, qp)
        # x = coordinate(coordinate_system_cache, qp)
        P, ∂P∂F = constitutive_driver(F, f₀, s₀, n₀, contraction_state, mp)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end


# ----------------------------------------------


#TODO Energy-based interface?
struct NormalSpringBC
    kₛ::Float64
    boundary_name::String
end

struct BendingSpringBC
    kᵇ::Float64
    boundary_name::String
end

struct ConstantPressureBC
    p::Float64
    boundary_name::String
end

struct SimpleFaceCache{MP, FV}
    mp::MP
    # time::Float64
    # microstructure_model::MM
    # coordinate_system::CS
    fv::FV
end

getboundaryname(face_cache::FC) where {FC} = face_cache.mp.boundary_name

function setup_face_cache(bcd::BCD, fv::FV) where {BCD, FV}
    SimpleFaceCache(bcd, fv)
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{NormalSpringBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack kₛ = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{BendingSpringBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            # Add contribution to the residual from this test function
            for i in 1:ndofs_face
                ∇δui = shape_gradient(fv, qp, i)
                residualₑ[i] += ∇δui ⊡ ∂Ψ∂F * dΓ

                ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
                for j in 1:ndofs_face
                    ∇δuj = shape_gradient(fv, qp, j)
                    # Add contribution to the tangent
                    Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
                end
            end
        end
    end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{ConstantPressureBC,FV}) where {FV}
    @unpack mp, fv = cache
    @unpack p = mp

    ndofs_face = getnbasefunctions(fv)
    for qp in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, qp)

        N = getnormal(fv, qp)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            # Add contribution to the residual from this test function
            for i in 1:ndofs_face
                δuᵢ = shape_value(fv, qp, i)
                residualₑ[i] += p * δuᵢ ⋅ N * dΓ
            end
        end
    end
end

function update_face_cache(cell::CC, face_cache::SimpleFaceCache{MP}) where {CC, MP}
end

# ----------------------------------------------


"""
TODO rewrite on per field basis.
"""
function assemble_global!(K::SparseMatrixCSC, f::Vector, dh::DH, cv::CV, fv::FV, material_model::MM, uₜ::Vector, uₜ₋₁::Vector, microstructure_model::MSM, calcium_field::CF, t::Float64, Δt::Float64, face_models::FM) where {DH, CV, FV, MM, MSM, CF, FM}
    n = ndofs_per_cell(dh)
    Kₑ = zeros(n, n)
    residualₑ = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    microstructure_cache = setup_microstructure_cache(cv, microstructure_model)
    # TODO this should be outside of this routine, because the contraction might be stateful! But where/how? Maybe all caches should be factored out...?
    contraction_cache = setup_contraction_model_cache(cv, material_model.contraction_model, calcium_field)
    element_cache = CardiacMechanicalElementCache(material_model, microstructure_cache, contraction_cache, cv)

    face_caches = ntuple(i->setup_face_cache(face_models[i], fv), length(face_models))

    # Loop over all cells in the grid
    # @timeit "assemble" for cell in CellIterator(dh)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)

        # TODO refactor
        uₑ = uₜ[global_dofs] # element dofs
        uₑ_prev = uₜ₋₁[global_dofs] # element dofs
        vₑ = (uₑ - uₑ_prev)/Δt # velocity approximation

        # Reinitialize cell values, and reset output arrays
        reinit!(cv, cell)
        fill!(Kₑ, 0.0)
        fill!(residualₑ, 0.0)

        # Update remaining caches specific to the element and models
        update_element_cache!(element_cache, calcium_field, t, cell)

        # Assemble matrix and residuals
        assemble_element!(Kₑ, residualₑ, uₑ, vₑ, element_cache)

        for local_face_index ∈ 1:nfaces(cell)
            face_is_initialized = false
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    if !face_is_initialized
                        face_is_initialized = true
                        reinit!(fv, cell, local_face_index)
                    end
                    update_face_cache(cell, face_cache)

                    assemble_face!(Kₑ, residualₑ, uₑ, face_cache)
                end
            end
        end

        assemble!(assembler, global_dofs, residualₑ, Kₑ)
    end
end


# ----------------------------------------------


function solve_test_ring(pv_name_base, material_model, grid, coordinate_system, microstructure_model, face_models, calcium_field, ip_mech::Interpolation{3, ref_shape}, ip_geo::Interpolation{3, ref_shape}, intorder, Δt = 0.1) where {ref_shape}
    io = ParaViewWriter(pv_name_base);

    T = 2.0

    # Finite element base
    qr = QuadratureRule{3, ref_shape}(intorder)
    qr_face = QuadratureRule{2, ref_shape}(intorder)
    cv = CellVectorValues(qr, ip_mech, ip_geo)
    fv = FaceVectorValues(qr_face, ip_mech, ip_geo)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    dbcs = ConstraintHandler(dh)
    # Clamp three sides
    dbc = Dirichlet(:u, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])
    add!(dbcs, dbc)
    # dbc = Dirichlet(:u, Set([first(getfaceset(grid, "Base"))]), (x,t) -> [0.0], [3])
    # add!(dbcs, dbc)
    # dbc = Dirichlet(:u, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    # add!(dbcs, dbc)

    close!(dbcs)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)

    uₜ   = zeros(_ndofs)
    uₜ₋₁ = zeros(_ndofs)
    Δu   = zeros(_ndofs)

    # ref_vol = calculate_volume_deformed_mesh(uₜ,dh,cv);
    # min_vol = ref_vol
    # max_vol = ref_vol

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    f = zeros(_ndofs)

    NEWTON_TOL = 1e-8
    MAX_NEWTON_ITER = 100

    for t ∈ 0.0:Δt:T
        # Store last solution
        uₜ₋₁ .= uₜ

        # Update with new boundary conditions (if available)
        Ferrite.update!(dbcs, t)
        apply!(uₜ, dbcs)

        # Perform Newton iterations
        newton_itr = -1
        while true
            newton_itr += 1

            assemble_global!(K, f, dh, cv, fv, material_model, uₜ, uₜ₋₁, microstructure_model, calcium_field, t, Δt, face_models)

            rhsnorm = norm(f[Ferrite.free_dofs(dbcs)])
            apply_zero!(K, f, dbcs)
            @info t rhsnorm

            if rhsnorm < NEWTON_TOL
                break
            elseif newton_itr > MAX_NEWTON_ITER
                error("Reached maximum Newton iterations. Aborting.")
            end

            #@info det(K)
            try
                Δu = K \ f
            catch
                finalize!(io)
                @warn "Failed Solve at " t
                return uₜ
            end

            apply_zero!(Δu, dbcs)

            uₜ .-= Δu # Current guess
        end

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

        microstructure_cache = setup_microstructure_cache(cv, microstructure_model)

        for cell in CellIterator(dh)
            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            uₑ = uₜ[global_dofs] # element dofs

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
                # @TODO compute properly
                v_longitudinal = Ferrite.Vec{3}((0.0, 0.0, 1.0))
                v_radial = Ferrite.Vec{3}((x_global[1],x_global[2],0.0))/norm(Ferrite.Vec{3}((x_global[1],x_global[2],0.0)))
                v_circimferential = Ferrite.Vec{3}((x_global[2],-x_global[1],0.0))/norm(Ferrite.Vec{3}((x_global[2],-x_global[1],0.0)))
                #
                E_ll_cell += v_longitudinal ⋅ E ⋅ v_longitudinal
                E_rr_cell += v_radial ⋅ E ⋅ v_radial
                E_cc_cell += v_circimferential ⋅ E ⋅ v_circimferential

                Jdata_cell += det(F)

                frefdata_cell += f₀
                srefdata_cell += s₀

                fdata_cell += f₀_current
                sdata_cell += s₀_current
            end

            E_ff[Ferrite.cellid(cell)] = E_ff_cell / nqp
            E_cc[Ferrite.cellid(cell)] = E_cc_cell / nqp
            E_rr[Ferrite.cellid(cell)] = E_rr_cell / nqp
            E_ll[Ferrite.cellid(cell)] = E_ll_cell / nqp
            Jdata[Ferrite.cellid(cell)] = Jdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] = frefdata_cell / nqp
            srefdata[Ferrite.cellid(cell)] = srefdata_cell / nqp
            fdata[Ferrite.cellid(cell)] = fdata_cell / nqp
            sdata[Ferrite.cellid(cell)] = sdata_cell / nqp
        end

        # Save the solution
        Thunderbolt.store_timestep!(io, t, dh, uₜ)
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
        Thunderbolt.finalize_timestep!(io, t)

        # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
        # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
    end

    # println("Compression: ", (ref_vol/min_vol - 1.0)*100, "%")
    # println("Expansion: ", (ref_vol/max_vol - 1.0)*100, "%")

    finalize!(io)

    return uₜ
end

using UnPack

Base.@kwdef struct NewActiveSpring{CPT}
    a::Float64   = 2.7
    aᶠ::Float64  = 1.6
    mpU::CPT = NullCompressionPenalty()
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring{CPT}) where {CPT}
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
    I₃ = det(C)
    J = det(F)
    I₁ = tr(C/cbrt(J^2))
    I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0)^2 + Thunderbolt.U(I₃, mpU)
end

Base.@kwdef struct NewHolzapfelOgden2009Model{TD,TU} #<: OrthotropicMaterialModel
    a₁::TD  =  5.0
    a₂::TD  =  2.5
    a₃::TD  =  0.5
    a₄::TD  = 10.0
    b₁::TD  =  5.0
    b₂::TD  =  2.5
    b₃::TD  =  0.5
    b₄::TD  =  2.0
    mpU::TU = SimpleCompressionPenalty(50.0)
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewHolzapfelOgden2009Model)
    @unpack a₁, b₁, a₂, b₂, a₃, b₃, a₄, b₄, mpU = mp

    C = tdot(F)
    I₃ = det(C)
    J = det(F)

    I₄ᶠ = f₀ ⋅ C ⋅ f₀ / cbrt(J^2)
    I₄ˢ = s₀ ⋅ C ⋅ s₀ / cbrt(J^2)
    I₄ⁿ = n₀ ⋅ C ⋅ n₀ / cbrt(J^2)
    I₈ᶠˢ = (f₀ ⋅ C ⋅ s₀ + s₀ ⋅ C ⋅ f₀)/2.0

    Ψᵖ = a₁/(2.0*b₁)*exp(b₁*(I₄ᶠ - 1.0)) + a₂/(2.0*b₂)*exp(b₂*(I₄ˢ -1.0)) + a₃/(2.0*b₃)*exp(b₃*(I₄ⁿ - 1.0)) + a₄/(2.0*b₄)*exp(b₄*(I₈ᶠˢ - 1.0)) + U(I₃, mpU)
    return Ψᵖ
end

# """
# Our reported fit against LinYin.
# """
Base.@kwdef struct NewActiveSpring2{CPT}
    # a::Float64   = 15.020986456784657
    # aᶠ::Float64  =  4.562365556553194
    a::Float64   = 2.6
    aᶠ::Float64  =  1.6
    mpU::CPT = NullCompressionPenalty()
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring2{CPT}) where {CPT}
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
    I₃ = det(C)
    J = det(F)
    I₁ = tr(C/cbrt(J^2))
    I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0) + Thunderbolt.U(I₃, mpU)
end

"""
"""
struct CalciumHatField end

"""
"""
value(coeff::CalciumHatField, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim} = t < 1.0 ? t : 2.0-t


using FerriteGmsh

for (filename, ref_shape, order) ∈ [
    # ("MidVentricularSectionQuadTet.msh", RefTetrahedron, 2),
    ("MidVentricularSectionTet.msh", RefTetrahedron, 1),
    # ("MidVentricularSectionHex.msh", RefCube, 1),
    # ("MidVentricularSectionQuadHex.msh", RefCube, 2) # We have to update FerriteGmsh first, because the hex27 translator is missing. See https://github.com/Ferrite-FEM/FerriteGmsh.jl/pull/29
]

ip_fiber = Lagrange{3, ref_shape, order}()
ip_geo = Lagrange{3, ref_shape, order}()

ring_grid = saved_file_to_grid("../data/meshes/ring/" * filename)
ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid, ip_geo)
ring_fm = create_simple_fiber_model(ring_cs, ip_fiber, ip_geo, endo_angle = 60.0, epi_angle = -70.0, endo_transversal_angle = -10.0, epi_transversal_angle = 20.0)

# passive_model = HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(4.0))

# solve_test_ring(filename*"_GHM-HO_AS1_GMKI_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         GMKIncompressibleActiveDeformationGradientModel(),
#         PelceSunLangeveld1995Model()
#     ), 
#     ring_grid, ring_cs, ring_fm,
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
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
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_AS1_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(NewActiveSpring()),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_GHM-HO_HO_RLRSQ75_Pelce",
#     GeneralizedHillModel(
#         passive_model,
#         ActiveMaterialAdapter(passive_model),
#         RLRSQActiveDeformationGradientModel(0.75),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Simple_Pelce",
#     ActiveStressModel(
#         passive_model,
#         SimpleActiveStress(),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

# solve_test_ring(filename*"_ActiveStress-HO_Piersanti_Pelce",
#     ActiveStressModel(
#         passive_model,
#         PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
#         PelceSunLangeveld1995Model()
#     ), ring_grid, ring_cs, ring_fm, 
#     [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
#     ip_geo, ip_geo, 2*order
# )

solve_test_ring(filename*"_GHM-HONEW_HONEW_RLRSQ100_Pelce",
    ExtendedHillModel(
        NewHolzapfelOgden2009Model(),
        ActiveMaterialAdapter(NewHolzapfelOgden2009Model(;mpU=NullCompressionPenalty())),
        RLRSQActiveDeformationGradientModel(0.5),
        PelceSunLangeveld1995Model()
    ), ring_grid, ring_cs, ring_fm, 
    [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
    ip_geo, ip_geo, 2*order
)

solve_test_ring(filename*"_ActiveStress-HONEW_Piersanti_Pelce",
    ActiveStressModel(
        NewHolzapfelOgden2009Model(),
        PiersantiActiveStress(2.0, 1.0, 0.75, 0.0),
        PelceSunLangeveld1995Model()
    ), ring_grid, ring_cs, ring_fm, 
    [NormalSpringBC(0.001, "Epicardium")], CalciumHatField(),
    ip_geo, ip_geo, 2*order
)

end
