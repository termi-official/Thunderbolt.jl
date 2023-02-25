using Thunderbolt
import Thunderbolt: Ψ, U

function assemble_element!(Kₑ, residualₑ, cell, cv, fv, mp, uₑ, uₑ_prev, fiber_model, time)
	# TODO factor out
	kₛ = 0.001 # "Spring stiffness"
	kᵇ = 0.0 # Basal bending penalty

	Caᵢ(id,x,t) = t < 1.0 ? t : 2.0-t
	#p = 7.5*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
	#p = 5.0*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
	# p = 0.0*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
    p = 0.0
	
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(Kₑ, 0.0)
    fill!(residualₑ, 0.0)

    ndofs = getnbasefunctions(cv)

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
		
        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u
		
        # Compute stress and tangent
		x_ref = cv.qr.points[qp]
		f₀, s₀, n₀ = directions(fiber_model, Ferrite.cellid(cell), x_ref)
		P, ∂P∂F = constitutive_driver(F, f₀, s₀, n₀, Caᵢ(Ferrite.cellid(cell), x_ref, time), mp)

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

    # Surface integrals
    for local_face_index in 1:nfaces(cell)
		# How does this interact with the stress?
		if (Ferrite.cellid(cell), local_face_index) ∈ getfaceset(cell.grid, "Epicardium")
            reinit!(fv, cell, local_face_index)
			ndofs_face = getnbasefunctions(fv)
            for qp in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, qp)

				∇u_prev = function_gradient(fv, qp, uₑ_prev)
                F_prev = one(∇u_prev) + ∇u_prev 
				N = transpose(inv(F_prev)) ⋅ getnormal(fv, qp) # TODO this may mess up reversibility

				# N = getnormal(fv, qp)
				
				u_q = function_value(fv, qp, uₑ)
				#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.0, u_q, :all)
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

				# N = getnormal(fv, qp)
				# u_q = function_value(fv, qp, uₑ)
				# for i ∈ 1:ndofs
				# 	δuᵢ = shape_value(fv, qp, i)
				# 	residualₑ[i] += 0.5 * kₛ * (δuᵢ ⋅ N) * (N ⋅ u_q) * dΓ
				# 	for j ∈ 1:ndofs
				# 		δuⱼ = shape_value(fv, qp, j)
				# 		Kₑ[i,j] += 0.5 * kₛ * (δuᵢ ⋅ N) * (N ⋅ δuⱼ) * dΓ
				# 	end
				# end
            end
        end

		if (Ferrite.cellid(cell), local_face_index) ∈ getfaceset(cell.grid, "Base")
			reinit!(fv, cell, local_face_index)
			ndofs_face = getnbasefunctions(fv)
            for qp in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, qp)
				N = getnormal(fv, qp)

				∇u = function_gradient(fv, qp, uₑ)
        		F = one(∇u) + ∇u

				#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(F_ -> 0.0, F, :all)
				∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

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

		# Pressure boundary
		if (Ferrite.cellid(cell), local_face_index) ∈ getfaceset(cell.grid, "Endocardium")
			reinit!(fv, cell, local_face_index)
			ndofs_face = getnbasefunctions(fv)
            for qp in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, qp)

				∇u_prev = function_gradient(fv, qp, uₑ_prev)
                F_prev = one(∇u_prev) + ∇u_prev 
				N = transpose(inv(F_prev)) ⋅ getnormal(fv, qp) # TODO this may mess up reversibility

				# N = getnormal(fv, qp)
				
				# u_q = function_value(fv, qp, uₑ)
				#∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.0, u_q, :all)
				# ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

				# Add contribution to the residual from this test function
				for i in 1:ndofs_face
		            δuᵢ = shape_value(fv, qp, i)
					residualₑ[i] += p * δuᵢ ⋅ N * dΓ

		            for j in 1:ndofs_face
		                δuⱼ = shape_value(fv, qp, j)
		                # Add contribution to the tangent
		                #Kₑ[i, j] += ( 0.0 ) * dΓ
		            end
				end
            end
		end
    end
end

"""
"""
function assemble_global!(K, f, dh, cv, fv, mp, uₜ, uₜ₋₁, fiber_model, t)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    #@timeit "assemble" for cell in CellIterator(dh)
	for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        uₑ = uₜ[global_dofs] # element dofs
        uₑ_prev = uₜ₋₁[global_dofs] # element dofs
		assemble_element!(ke, ge, cell, cv, fv, mp, uₑ, uₑ_prev, fiber_model, t)
        assemble!(assembler, global_dofs, ge, ke)
    end
end

function solve_test_ring(pv_name_base, mp, grid, coordinate_system, fiber_model)
	# geo = Tetrahedron
	# refgeo = RefTetrahedron
	# intorder = 1

	geo = Hexahedron
	refgeo = RefCube
	order = 1
	intorder = 2

	pvd = paraview_collection("$pv_name_base.pvd");

	T = 2.0
	Δt = 0.1

    # Material parameters
    # E = 4.0
    # ν = 0.45
    # μ = E / (2(1 + ν))
    # λ = (E * ν) / ((1 + ν) * (1 - 2ν))
	# μ = 0.001
	# λ = 0.001
	# η = 1.001
    # mp = ActiveNeoHookean(μ, λ, η)
	#mp = Passive2017Energy(1.0, 2.6, 2.82, 2.0, 30.48, 7.25, 1.0, 100.0)
	#mp = BioNeoHooekan(4.0, 10.25, 1, 2, 10.0)
	#mp = BioNeoHooekan(1.01, 1.01, 1, 2, 10.0)
	# mp = BioNeoHooekan(0.25, 4.00, 1, 2, 15.0) # Looks okay :)
	#mp = HolzapfelOgden2009(η = 15.0)
	#mp = AmborsiPezzuto2014(A=0.25,B=0.0,K=5.0)

    # Finite element base
    ip = Lagrange{3, refgeo, order}()
    ip_geo = Lagrange{3, refgeo, order}()
    qr = QuadratureRule{3, refgeo}(intorder)
    qr_face = QuadratureRule{2, refgeo}(intorder)
    cv = CellVectorValues(qr, ip, ip_geo)
    fv = FaceVectorValues(qr_face, ip, ip_geo)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

	vtk_save(vtk_grid("$pv_name_base-grid.vtu", dh))

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

	ref_vol = calculate_volume_deformed_mesh(uₜ,dh,cv);
	min_vol = ref_vol
	max_vol = ref_vol

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)

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

	        assemble_global!(K, g, dh, cv, fv, mp, uₜ, uₜ₋₁, fiber_model, t)
	        normg = norm(g[Ferrite.free_dofs(dbcs)])
	        apply_zero!(K, g, dbcs)
			@info "t = " t ": ||g|| = " normg
	
	        if normg < NEWTON_TOL
	            break
	        elseif newton_itr > MAX_NEWTON_ITER
	            error("Reached maximum Newton iterations. Aborting.")
	        end

			#@info det(K)
			Δu = K \ g
	
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

		frefdata = zero(Vector{Vec{3}}(undef, getncells(grid)))
		srefdata = zero(Vector{Vec{3}}(undef, getncells(grid)))
		fdata = zero(Vector{Vec{3}}(undef, getncells(grid)))
		sdata = zero(Vector{Vec{3}}(undef, getncells(grid)))

		for cell in CellIterator(dh)
			reinit!(cv, cell)
			global_dofs = celldofs(cell)
        	uₑ = uₜ[global_dofs] # element dofs
			
			E_ff_cell = 0.0
			E_ff_cell2 = 0.0
			E_cc_cell = 0.0
			E_rr_cell = 0.0
			E_ll_cell = 0.0

			Jdata_cell = 0.0
			frefdata_cell = Vec{3}((0.0, 0.0, 0.0))
			srefdata_cell = Vec{3}((0.0, 0.0, 0.0))
			fdata_cell = Vec{3}((0.0, 0.0, 0.0))
			sdata_cell = Vec{3}((0.0, 0.0, 0.0))

			nqp = getnquadpoints(cv)
			for qp in 1:nqp
		        dΩ = getdetJdV(cv, qp)

		        # Compute deformation gradient F
		        ∇u = function_gradient(cv, qp, uₑ)
		        F = one(∇u) + ∇u

				C = tdot(F)
				E = (C-one(C))/2.0
				x_ref = cv.qr.points[qp]
				f₀,s₀,n₀ = directions(fiber_model, Ferrite.cellid(cell), x_ref)

				E_ff_cell += f₀ ⋅ E ⋅ f₀
				
				f₀_current = F⋅f₀
				f₀_current /= norm(f₀_current)

				s₀_current = F⋅s₀
				s₀_current /= norm(s₀_current)

				E_ff_cell2 += f₀_current ⋅ E ⋅ f₀_current

				coords = getcoordinates(cell)
				x_global = spatial_coordinate(cv, qp, coords)
				# @TODO compute properly
				v_longitudinal = Vec{3}((0.0, 0.0, 1.0))
				v_radial = Vec{3}((x_global[1],x_global[2],0.0))/norm(Vec{3}((x_global[1],x_global[2],0.0)))
				v_circimferential = Vec{3}((x_global[2],-x_global[1],0.0))/norm(Vec{3}((x_global[2],-x_global[1],0.0)))
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
			E_ff2[Ferrite.cellid(cell)] = E_ff_cell2 / nqp
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
		vtk_grid("$pv_name_base-$t.vtu", dh) do vtk
            vtk_point_data(vtk,dh,uₜ)
	        vtk_cell_data(vtk,hcat(frefdata...),"Reference Fiber Data")
	        vtk_cell_data(vtk,hcat(srefdata...),"Reference Sheet Data")
	        vtk_cell_data(vtk,hcat(fdata...),"Current Fiber Data")
			vtk_cell_data(vtk,hcat(sdata...),"Current Sheet Data")
	        vtk_cell_data(vtk,E_ff,"E_ff")
	        vtk_cell_data(vtk,E_ff2,"E_ff2")
	        vtk_cell_data(vtk,E_cc,"E_cc")
	        vtk_cell_data(vtk,E_rr,"E_rr")
	        vtk_cell_data(vtk,E_ll,"E_ll")
	        vtk_cell_data(vtk,Jdata,"J")
            vtk_save(vtk)
	        pvd[t] = vtk
	    end

		min_vol = min(min_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
		max_vol = max(max_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
	end

	println("Compression: ", (ref_vol/min_vol - 1.0)*100, "%")
	println("Expansion: ", (ref_vol/max_vol - 1.0)*100, "%")
	
	vtk_save(pvd);

	return uₜ
end

LVS_grid = generate_ring_mesh(30,4,3; longitudinal_upper=0.25)
LVS_cs = compute_midmyocardial_section_coordinate_system(LVS_grid)
LVS_fm = create_simple_fiber_model(LVS_cs, Lagrange{3, RefCube, 1}(), endo_angle = 60.0, epi_angle = -60.0, endo_transversal_angle = -20.0, epi_transversal_angle = 20.0)

"""
Our fit against LinYin.
"""
Base.@kwdef struct NewActiveSpring2
	a   = 15.020986456784657
	aᶠ  =  4.562365556553194
	mpU = NullCompressionModel()
end

function Thunderbolt.Ψ(F, f₀, s₀, n₀, mp::NewActiveSpring2)
    @unpack a, aᶠ, mpU = mp

    C = tdot(F)
	I₃ = det(C)
	J = det(F)
    I₁ = tr(C/cbrt(J^2))
	I₄ᶠ = f₀ ⋅ C ⋅ f₀

    return a/2.0*(I₁-3.0)^2 + aᶠ/2.0*(I₄ᶠ-1.0) + Thunderbolt.U(I₃, mpU)
end

solve_test_ring("HO-LVS", 
    GeneralizedHillModel(
        HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(1.0)),
        ActiveMaterialAdapter(NewActiveSpring2()),
        RLRSQActiveDeformationGradientModel(1.0),
        PelceSunLangeveld1995Model()
    ), LVS_grid, LVS_cs, LVS_fm
)

solve_test_ring("HO-AS", 
    ActiveStressModel(
        HolzapfelOgden2009Model(1.5806251396691438, 5.8010248271289395, 0.28504197825657906, 4.126552003938297, 0.0, 1.0, 0.0, 1.0, SimpleCompressionPenalty(10.0)),
        PiersantiActiveStress(3.0, 1.0, 1.0, 0.0),
        PelceSunLangeveld1995Model()
    ), LVS_grid, LVS_cs, LVS_fm
)
