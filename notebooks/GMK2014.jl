### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 48685583-8831-4797-bde9-f2032ef42837
let
	using Pkg
	Pkg.activate(Base.current_project());
	
	using Ferrite, FerriteGmsh, Tensors, Thunderbolt, Optim, UnPack, Plots, LinearAlgebra
end

# ╔═╡ 6e43f86d-6341-445f-be8d-146eb0447457
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 12%);
    	padding-right: max(160px, 12%);
	}
</style>
"""

# ╔═╡ f6b5bde0-bcdc-41de-b64e-29b9e60b06c0
md"""
# The generalized Hill model: A kinematic approach towards active muscle contraction

## Framework Recap

For this summary and implementation we assume a cartesian coordinate system and an imposed calcium field.

We start by introducing the multiplicative decomposition of the deformation gradient $\mathbf{F} = \mathbf{F}^{\mathrm{e}} \mathbf{F}^{\mathrm{a}}$, assuming that the active component creates a plasticity-like intermediate configuration. Since $\mathbf{F}$ must be invertible way we can express the elastic portion of the deformation gradient via $\mathbf{F}^{\mathrm{e}} = \mathbf{F} {\mathbf{F}^\mathrm{a}}^{-1}$. $\mathbf{F}^{\mathrm{a}}$ is assumed to be computable from the active contraction model. In this paper $\mathbf{F}^{\mathrm{a}}$ is defined as

```math
\mathbf{F}^{\mathrm{a}} = \mathbf{I} + (\lambda^{\mathrm{a}}_{\mathbf{f}_0} - 1) \mathbf{f}_0 \otimes \mathbf{f}_0 \, ,
```

where the active stretch $\lambda^{\mathrm{a}}_{\mathbf{f}_0}$ is modeled as a function of the normalized intracellular calcium concentration $\overline{[\mathrm{Ca}]_{\mathrm{i}}}$ via 

```math
\lambda^{\mathrm{a}}_{\mathbf{f}_0}(\overline{[\mathrm{Ca}]_{\mathrm{i}}}) = \frac{\xi(\overline{[\mathrm{Ca}]_{\mathrm{i}}})}{1+\chi(\overline{[\mathrm{Ca}]_{\mathrm{i}}})(\xi(\overline{[\mathrm{Ca}]_{\mathrm{i}}})-1)}\lambda^{\mathrm{a,max}}_{\mathbf{f}_0}\, ,
```

which translates to the issue that contraction velocity is not included into the model. We can also interpret his model as having an instantaneous response.

Consequently we can express the elastic portion of the deformation gradient as 

```math
\mathbf{F}^{\mathbf{e}} = \mathbf{F} - (1-\lambda^{\mathrm{a}}_{\mathbf{f}_0}(\overline{[\mathrm{Ca}]_{\mathrm{i}}})^{-1}) \mathbf{F} \mathbf{f}_0 \otimes \mathbf{f}_0 \, .
```

For the springs we have two Helmholtz free energy density functions, namely $\Psi^{\mathrm{p}}(\mathbf{F})$ for the "passive" spring and the serial spring attached to the active element $\Psi^{\mathrm{s}}(\mathbf{F}^{\mathbf{e}})$. Note that $\mathbf{F}^{\mathrm{e}}$ is not explicitly given, but we can recover it as discussed above, giving us $\Psi^{\mathrm{s}}(\mathbf{F}{\mathbf{F}^{\mathrm{a}}}^{-1}) \equiv \Psi^{\mathrm{s}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}})$. We summarize these potentials under $\Psi^{\mathrm{spring}}(\mathbf{F},\mathbf{F}^{\mathrm{a}})$. Note that the serial spring is chosen to be linear in the paper.

With this information we can express the Piola-Kirchhoff stress tensor as

```math
\mathbf{P}
= \partial_{\mathbf{F}} \Psi^{\mathrm{spring}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}}) 
= \partial_{\mathbf{F}} \Psi^{\mathrm{p}}(\mathbf{F}) + \partial_{\mathbf{F}} \Psi^{\mathrm{s}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}}) \, .
```



"""

# ╔═╡ 94fc407e-d72e-4b44-9d30-d66c433b954a
md"""
## Idealized Left Ventricle
"""

# ╔═╡ a81fc16e-c280-4b38-8c6a-18714db7840c
function get_local_face_dofs(field_name, components, dh::DofHandler, local_face_index)
	field_idx = Ferrite.find_field(dh, field_name)

    # Extract stuff for the field
    interpolation = Ferrite.getfieldinterpolation(dh, field_idx)
    field_dim = Ferrite.getfielddim(dh, field_idx)
	offset = Ferrite.field_offset(dh, field_name)
	
	local_face_dofs = Int[]
	for (_, face) in enumerate(Ferrite.faces(interpolation)[local_face_index])
        for fdof in face, d in 1:field_dim
            if d in components
                push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
            end
        end
    end
    return local_face_dofs
end

# ╔═╡ d314acc8-22fc-40f8-bcc6-729b6e208f70
struct NeoHookeanModel
	λ
	μ
end

# ╔═╡ bc3bbb45-c263-4f29-aa6f-647e8dbf3466
function ψ(F, Mᶠ, model::NeoHookeanModel)
	@unpack μ, λ = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	I₃ = det(C)
	J = sqrt(I₃)

    return μ / 2 * (I₁ - 3) - μ * log(J) + λ / 2 * log(J)^2
end

# ╔═╡ f4a00f5d-f042-4804-9be1-d24d5046fd0a
struct FiberSheetNormal
	f
	s
	n
end

# ╔═╡ 610c857e-a699-48f6-b18b-df8337287122
grid = saved_file_to_grid("../data/meshes/EllipsoidalLeftVentricle.msh")

# ╔═╡ 9e599448-f844-40c2-b237-2820138aebe0
md"""
## Simple Coordinate System
"""

# ╔═╡ 2ec1e3de-0a48-431f-b3b9-ee9ec1390504
struct LVCoordinateSystem
	dh
	cellvalues
	apicobasal
	transmural
end

# ╔═╡ fa085581-aea6-4c80-8b21-ac59c9ba8fd0
function compute_LV_coordinate_system(grid)
	ip = Lagrange{3, RefTetrahedron, 1}()
	qr = QuadratureRule{3, RefTetrahedron}(2)
	cellvalues = CellScalarValues(qr, ip);

	dh = DofHandler(grid)
	push!(dh, :coordinates, 1)
	vertex_dict,_,_,_ = Ferrite.__close!(dh);

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

	vtk_grid("coordinates_transmural", dh) do vtk
	    vtk_point_data(vtk, dh, transmural)
	end

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

	vtk_grid("coordinates_apicobasal", dh) do vtk
	    vtk_point_data(vtk, dh, apicobasal)
	end

	return LVCoordinateSystem(dh, cellvalues, transmural, apicobasal)
end

# ╔═╡ 793a52db-3f7b-4c5b-984a-f7ac6e9fc402
coordinate_system = compute_LV_coordinate_system(grid)

# ╔═╡ 0e5198da-d533-4a64-8dd0-46ba2418f099
function extract_epicardial_edges(grid)
	
end

# ╔═╡ b34e1ab1-8c27-47a2-a2ba-be9956714ffe
md"""
## Simple Fiber Model
"""

# ╔═╡ 9f58b6cf-8b9f-4b48-bc35-180ab9559d3d
struct PiecewiseConstantFiberModel
	f₀data
	s₀data
	n₀data
end

# ╔═╡ 3337cac5-2a53-4043-9751-1ffc6ee37889
function f₀(fiber_model::PiecewiseConstantFiberModel, cell_id, x_ref)
	return fiber_model.f₀data[cell_id]
end

# ╔═╡ 0f55501e-694b-40ba-be6c-faab5f6177b5
# Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.
function create_simple_pw_constant_fiber_model(coordinate_system)
	dh = coordinate_system.dh
	f₀data = Vector{Vec{3}}(undef, getncells(dh.grid))
	ip = dh.field_interpolations[1] #TODO refactor this. Pls.
	qr = QuadratureRule{3,RefTetrahedron,Float64}([1.0], [Vec{3}((0.1, 0.1, 0.1))]) #TODO is this really we want to do?
	cv = CellScalarValues(qr, ip)
	for (cellindex,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
		dof_indices = celldofs(cell)

		∇apicobasal = function_gradient(cv, 1, coordinate_system.apicobasal[dof_indices])
		∇transmural = function_gradient(cv, 1, coordinate_system.transmural[dof_indices])

		f₀data[cellindex] = cross(∇apicobasal, ∇transmural)
		f₀data[cellindex] /= norm(f₀data[cellindex])
	end
	
	PiecewiseConstantFiberModel(f₀data, [], [])
end

# ╔═╡ b8a970e3-3d7f-4ed8-a546-c26e9047b443
fiber_model = create_simple_pw_constant_fiber_model(coordinate_system)

# ╔═╡ 620c34e6-48b0-49cf-8b3f-818107d0bc94
md"""
## Driver Code
"""

# ╔═╡ 2eae0e78-aba6-49f5-84b1-dff9c087b391
struct NeoHookean
    μ::Float64
    λ::Float64
end

# ╔═╡ 0658ec98-5883-4859-93dc-f9e9f52f9f63
function Ψ(C, mp::NeoHookean)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

# ╔═╡ b8c36e6f-862a-4ade-afee-b1cdb40a4ed7
λᵃₘₐₓ = 0.7

# ╔═╡ c7f78dd3-f75e-4634-bee9-14e6efe28e68
λᵃβ = 3.0

# ╔═╡ 374329cc-dcd5-407b-a4f2-83f21120577f
#λᵃ(Caᵢ) = (cos(pi*x)*(1-λᵃₘₐₓ) + 1.0)/2.0 + λᵃₘₐₓ/2.0
λᵃ(Caᵢ) = 1.0/(1+(0.5+atan(λᵃβ*log(max(Caᵢ,1e-10)))/π))

# ╔═╡ 9d9da356-ac29-4a99-9a1a-e8f61141a8d1
struct ActiveNeoHookean
    μ::Float64
    λ::Float64
	η::Float64
end

# ╔═╡ 03dbf71b-c69a-4049-ad2f-1f78ae754fde
function Ψ(F, f₀, Caᵢ, mp::ActiveNeoHookean)
	C = tdot(F)
    μ = mp.μ
    λ = mp.λ
	η = mp.η
    Ic = tr(C)
    J = det(F)

	Ψᵖ = μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2 

	M = f₀ ⊗ f₀
	Fᵃ = one(F) + (λᵃ(Caᵢ) - 1.0) * M
	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
	M̃ = f̃ ⊗ f̃

	Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀)
	Iᵉ₄ = tr(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))
	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2
	
    return Ψᵖ + Ψᵃ
end

# ╔═╡ 4ff78cdf-1efc-4c00-91a3-4c29f3d27305
function constitutive_driver(C, mp::NeoHookean)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 4.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

# ╔═╡ 641ad832-1b22-44d0-84f0-bfe15ecd6246
function constitutive_driver(F, f₀, Caᵢ, mp::ActiveNeoHookean)
    # Compute all derivatives in one function call
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> Ψ(y, f₀, Caᵢ, mp), F, :all)
    return ∂Ψ∂F, ∂²Ψ∂F²
end;

# ╔═╡ b2b670d9-2fd7-4031-96bb-167db12475c7
function assemble_element!(cellid, Kₑ, residuumₑ, cell, cv, fv, mp, uₑ, fiber_model, time)
	# TODO factor out
	kₛ = 10.0 # "Spring stiffness"
	Caᵢ(cellid,x,t) = t
	
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(Kₑ, 0.0)
    fill!(residuumₑ, 0.0)

    traction = Vec{3}((0.0, 0.0, 0.0)) # Traction
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
		
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u
		
        # Compute stress and tangent
		#X = spatial_coordinate(cv, qp, getcoordinates(cell))
		#TODO compute coordinate
		x_ref = Vec{3}((0.0, 0.0, 0.0))
		P, ∂P∂F = constitutive_driver(F, f₀(fiber_model, cellid, x_ref), Caᵢ(cellid, x_ref, time), mp)

        # Loop over test functions
        for i in 1:ndofs
            # Test function + gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
			
            # Add contribution to the residual from this test function
            residuumₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end

    # Surface integral for the traction
    for local_face_index in 1:nfaces(cell)
		if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Endocardium")
            reinit!(fv, cell, local_face_index)
            for q_point in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    residuumₑ[i] -= (δui ⋅ traction) * dΓ
                end
            end
        end

		if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Epicardium")
            reinit!(fv, cell, local_face_index)
            for qp in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, qp)
				N = getnormal(fv, qp)
				for i ∈ 1:ndofs
					δuᵢ = shape_value(fv, qp, i)
					for j ∈ 1:ndofs
						δuⱼ = shape_value(fv, qp, j)
						Kₑ[i,j] += kₛ * (δuᵢ ⋅ N) * (N ⋅ δuⱼ) * dΓ
					end
				end
            end
        end
    end
end;

# ╔═╡ aa57fc70-16f4-4013-a71e-a4099f0e5bd4
function assemble_global!(K, f, dh, cv, fv, mp, u, fiber_model, t)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    #@timeit "assemble" for cell in CellIterator(dh)
	for (cellid,cell) in enumerate(CellIterator(dh))
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
		assemble_element!(cellid, ke, ge, cell, cv, fv, mp, ue, fiber_model, t)
        assemble!(assembler, global_dofs, ge, ke)
    end
end;

# ╔═╡ edb7ba53-ba21-4001-935f-ec9e0d7531be
function solve(grid, fiber_model)
	pvd = paraview_collection("GMK2014_LV.pvd");
	
    # Material parameters
    E = 10.0
    ν = 0.3
	η = 10.0
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = ActiveNeoHookean(μ, λ, η)

    # Finite element base
    ip = Lagrange{3, RefTetrahedron, 1}()
    qr = QuadratureRule{3, RefTetrahedron}(1)
    qr_face = QuadratureRule{2, RefTetrahedron}(1)
    cv = CellVectorValues(qr, ip)
    fv = FaceVectorValues(qr_face, ip)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    function rotation(X, t, θ = deg2rad(60.0))
        x, y, z = X
        return t * Vec{3}(
            (0.0,
            L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ),
            L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ)
            ))
    end

    dbcs = ConstraintHandler(dh)
    # Clamp base for now
    #dbc = Dirichlet(:u, getfaceset(grid, "Base"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    #add!(dbcs, dbc)
	
    close!(dbcs)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    apply!(u, dbcs)

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)

    NEWTON_TOL = 1e-6
	MAX_NEWTON_ITER = 100

	for t ∈ 0.0:0.1:1.0
		@info "t = " t

	    Ferrite.update!(dbcs, t)
		# Reset initial guess...
		u .= 0.0
	    apply!(u, dbcs)

		# Perform Newton iterations
		newton_itr = -1
	    while true
			newton_itr += 1
			
	        u .-= Δu # Current guess
	        assemble_global!(K, g, dh, cv, fv, mp, u, fiber_model, t)
	        normg = norm(g[Ferrite.free_dofs(dbcs)])
	        apply_zero!(K, g, dbcs)
			@info "||g|| = " normg
	
	        if normg < NEWTON_TOL
	            break
	        elseif newton_itr > MAX_NEWTON_ITER
	            error("Reached maximum Newton iterations. Aborting.")
	        end
	
	        # Compute increment using cg! from IterativeSolvers.jl
			# Δu, flag, relres, iter, resvec = KrylovMethods.cg(K, g; maxIter = 1000)
	
			# Nope. Opt for direct solver ..for now. :)
			Δu = K \ g
	
	        apply_zero!(Δu, dbcs)
	    end
	
	    # Save the solution
		vtk_grid("GMK2014-LV-$t.vtu", dh) do vtk
            vtk_point_data(vtk,dh,u)
	        vtk_cell_data(vtk,hcat(fiber_model.f₀data...),"Reference Fiber Data")
            vtk_save(vtk)
	        pvd[t] = vtk
	    end
	end
	
	vtk_save(pvd);

	return u
end

# ╔═╡ 8cfeddaa-c67f-4de8-b81c-4fbb7e052c50
solve(grid, fiber_model)

# ╔═╡ Cell order:
# ╟─6e43f86d-6341-445f-be8d-146eb0447457
# ╟─48685583-8831-4797-bde9-f2032ef42837
# ╟─f6b5bde0-bcdc-41de-b64e-29b9e60b06c0
# ╟─94fc407e-d72e-4b44-9d30-d66c433b954a
# ╟─a81fc16e-c280-4b38-8c6a-18714db7840c
# ╠═d314acc8-22fc-40f8-bcc6-729b6e208f70
# ╠═bc3bbb45-c263-4f29-aa6f-647e8dbf3466
# ╟─f4a00f5d-f042-4804-9be1-d24d5046fd0a
# ╟─610c857e-a699-48f6-b18b-df8337287122
# ╟─9e599448-f844-40c2-b237-2820138aebe0
# ╠═2ec1e3de-0a48-431f-b3b9-ee9ec1390504
# ╠═fa085581-aea6-4c80-8b21-ac59c9ba8fd0
# ╠═793a52db-3f7b-4c5b-984a-f7ac6e9fc402
# ╟─0e5198da-d533-4a64-8dd0-46ba2418f099
# ╟─b34e1ab1-8c27-47a2-a2ba-be9956714ffe
# ╠═9f58b6cf-8b9f-4b48-bc35-180ab9559d3d
# ╠═3337cac5-2a53-4043-9751-1ffc6ee37889
# ╠═0f55501e-694b-40ba-be6c-faab5f6177b5
# ╠═b8a970e3-3d7f-4ed8-a546-c26e9047b443
# ╟─620c34e6-48b0-49cf-8b3f-818107d0bc94
# ╠═2eae0e78-aba6-49f5-84b1-dff9c087b391
# ╟─0658ec98-5883-4859-93dc-f9e9f52f9f63
# ╠═b8c36e6f-862a-4ade-afee-b1cdb40a4ed7
# ╠═c7f78dd3-f75e-4634-bee9-14e6efe28e68
# ╠═374329cc-dcd5-407b-a4f2-83f21120577f
# ╟─4ff78cdf-1efc-4c00-91a3-4c29f3d27305
# ╠═9d9da356-ac29-4a99-9a1a-e8f61141a8d1
# ╠═03dbf71b-c69a-4049-ad2f-1f78ae754fde
# ╠═641ad832-1b22-44d0-84f0-bfe15ecd6246
# ╠═b2b670d9-2fd7-4031-96bb-167db12475c7
# ╠═aa57fc70-16f4-4013-a71e-a4099f0e5bd4
# ╠═edb7ba53-ba21-4001-935f-ec9e0d7531be
# ╠═8cfeddaa-c67f-4de8-b81c-4fbb7e052c50
