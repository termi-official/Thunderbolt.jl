### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 48685583-8831-4797-bde9-f2032ef42837
let
	using Pkg
	Pkg.activate(Base.current_project());
	
	using Ferrite, FerriteGmsh, Tensors, Thunderbolt, Optim, UnPack, Plots
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

# ╔═╡ 0e5198da-d533-4a64-8dd0-46ba2418f099
function extract_epicardial_edges(grid)
	
end

# ╔═╡ 2eae0e78-aba6-49f5-84b1-dff9c087b391
struct NeoHooke
    μ::Float64
    λ::Float64
end

# ╔═╡ 0658ec98-5883-4859-93dc-f9e9f52f9f63
function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

# ╔═╡ 4ff78cdf-1efc-4c00-91a3-4c29f3d27305
function constitutive_driver(C, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 4.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

# ╔═╡ b2b670d9-2fd7-4031-96bb-167db12475c7
function assemble_element!(Kₑ, residuumₑ, cell, cv, fv, mp, uₑ)
	kₛ = 1.0 # "Spring stiffness"
	
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(Kₑ, 0.0)
    fill!(residuumₑ, 0.0)

    t = Vec{3}((0.0, 0.0, -1.0)) # Traction
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
		
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u
        C = tdot(F)
		
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I) + otimesu(I, S)

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
                    residuumₑ[i] -= (δui ⋅ t) * dΓ
                end
            end
        end

		if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Epicardium")
            reinit!(fv, cell, local_face_index)
			facedofs = get_local_face_dofs(:u, [1,2,3], cell.dh, local_face_index)
			
            for q_point in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
					for j in 1:ndofs
					end
                    #residuumₑ[i] -= (δui ⋅ ui) * dΓ
                end
            end
        end
    end
end;

# ╔═╡ aa57fc70-16f4-4013-a71e-a4099f0e5bd4
function assemble_global!(K, f, dh, cv, fv, mp, u)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    #@timeit "assemble" for cell in CellIterator(dh)
	for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        #@timeit "element assemble" assemble_element!(ke, ge, cell, cv, fv, mp, ue)
		assemble_element!(ke, ge, cell, cv, fv, mp, ue)
        assemble!(assembler, global_dofs, ge, ke)
    end
end;

# ╔═╡ edb7ba53-ba21-4001-935f-ec9e0d7531be
function solve(grid)
    #reset_timer!()

    # Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

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
    dbc = Dirichlet(:u, getfaceset(grid, "Base"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
	
    close!(dbcs)
    t = 0.5
    Ferrite.update!(dbcs, t)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    apply!(u, dbcs)

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)


    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1e-8
    #prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")

    while true; newton_itr += 1
        u .-= Δu # Current guess
        assemble_global!(K, g, dh, cv, fv, mp, u)
        normg = norm(g[Ferrite.free_dofs(dbcs)])
        apply_zero!(K, g, dbcs)
        #ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])

        if normg < NEWTON_TOL
            break
        elseif newton_itr > 30
            error("Reached maximum Newton iterations, aborting")
        end

        # Compute increment using cg! from IterativeSolvers.jl
		# Δu, flag, relres, iter, resvec = KrylovMethods.cg(K, g; maxIter = 1000)
		Δu = K \ g

        apply_zero!(Δu, dbcs)
    end

    # Save the solution
    #@timeit "export" begin
        vtk_grid("hyperelasticity", dh) do vtkfile
            vtk_point_data(vtkfile, dh, u)
        end
    #end

    #print_timer(title = "Analysis with $(getncells(grid)) elements", linechars = :ascii)
    return u
end

# ╔═╡ 8cfeddaa-c67f-4de8-b81c-4fbb7e052c50
solve(grid)

# ╔═╡ Cell order:
# ╟─6e43f86d-6341-445f-be8d-146eb0447457
# ╟─48685583-8831-4797-bde9-f2032ef42837
# ╟─f6b5bde0-bcdc-41de-b64e-29b9e60b06c0
# ╟─94fc407e-d72e-4b44-9d30-d66c433b954a
# ╟─a81fc16e-c280-4b38-8c6a-18714db7840c
# ╟─d314acc8-22fc-40f8-bcc6-729b6e208f70
# ╟─bc3bbb45-c263-4f29-aa6f-647e8dbf3466
# ╟─f4a00f5d-f042-4804-9be1-d24d5046fd0a
# ╟─610c857e-a699-48f6-b18b-df8337287122
# ╠═0e5198da-d533-4a64-8dd0-46ba2418f099
# ╠═2eae0e78-aba6-49f5-84b1-dff9c087b391
# ╠═0658ec98-5883-4859-93dc-f9e9f52f9f63
# ╠═4ff78cdf-1efc-4c00-91a3-4c29f3d27305
# ╠═b2b670d9-2fd7-4031-96bb-167db12475c7
# ╠═aa57fc70-16f4-4013-a71e-a4099f0e5bd4
# ╠═edb7ba53-ba21-4001-935f-ec9e0d7531be
# ╠═8cfeddaa-c67f-4de8-b81c-4fbb7e052c50
