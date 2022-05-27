### A Pluto.jl notebook ###
# v0.19.5

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
	qr
	cellvalues
	transmural
	apicobasal
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

	return LVCoordinateSystem(dh, qr, cellvalues, transmural, apicobasal)
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
	qr = QuadratureRule{3,RefTetrahedron,Float64}([1.0], [Vec{3}((0.25, 0.25, 0.25))]) #TODO is this really we want to do?
	cv = CellScalarValues(qr, ip)
	for (cellindex,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
		dof_indices = celldofs(cell)

		# compute fiber direction
		∇apicobasal = function_gradient(cv, 1, coordinate_system.apicobasal[dof_indices])
		∇transmural = function_gradient(cv, 1, coordinate_system.transmural[dof_indices])
		v = ∇apicobasal × ∇transmural

		transmural  = function_value(cv, 1, coordinate_system.transmural[dof_indices])

		# linear interpolation of rotation angle
		endo_angle = 80.0
		epi_angle  = -50.0
		θ = (1-transmural) * endo_angle + (transmural) * epi_angle

		# Rodriguez rotation
		sinθ = sin(deg2rad(θ))
		cosθ = cos(deg2rad(θ))
		k = ∇transmural / norm(∇transmural)
		vᵣ = v * cosθ + (k × v) * sinθ + k * (k ⋅ v) * (1-cosθ)

		f₀data[cellindex] = vᵣ / norm(vᵣ)
	end

	PiecewiseConstantFiberModel(f₀data, [], [])
end

# ╔═╡ b8a970e3-3d7f-4ed8-a546-c26e9047b443
fiber_model = create_simple_pw_constant_fiber_model(coordinate_system)

# ╔═╡ aaf500d1-c0ef-4839-962a-139503e6fcc0
struct FieldCoefficient
	elementwise_data #3d array (element_idx, base_fun_idx, dim)
	ip
end

# ╔═╡ dedc939c-b789-4a8e-bc5e-e70e40bb6b8d
function f₀(coeff::FieldCoefficient, cell_id, ξ)
	@unpack ip, elementwise_data = coeff
	dim = 3 #@FIXME PLS

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{dim, Float64})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.value(ip, i, ξ) * elementwise_data[cell_id, i]
    end
    return val / norm(val)
end

# ╔═╡ 0c863dcc-c5aa-47a7-9941-97b7a392febd
function generate_nodal_quadrature_rule(ip::Interpolation{dim, ref_shape, order}) where {dim, ref_shape, order}
	n_base = Ferrite.getnbasefunctions(ip)
	positions = Ferrite.reference_coordinates(ip)
	return QuadratureRule{dim, ref_shape, Float64}(ones(length(positions)), positions)
end

# ╔═╡ cb09e0f9-fd04-4cb9-bed3-a1e97cb6d478
# Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.
function create_simple_fiber_model(coordinate_system, ip_fiber; endo_angle = 80.0, epi_angle = -65.0, endo_transversal_angle = 0.0, epi_transversal_angle = 0.0)
	@unpack dh = coordinate_system
	
	ip = dh.field_interpolations[1] #TODO refactor this. Pls.

	n_basefuns = getnbasefunctions(ip_fiber)
	dim = 3

	elementwise_data = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))

	qr_fiber = generate_nodal_quadrature_rule(ip_fiber)
	cv = CellScalarValues(qr_fiber, ip)

	for (cellindex,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
		dof_indices = celldofs(cell)

		for qp in 1:getnquadpoints(cv)
			# compute fiber direction
			∇apicobasal = function_gradient(cv, qp, coordinate_system.apicobasal[dof_indices])
			∇transmural = function_gradient(cv, qp, coordinate_system.transmural[dof_indices])
			∇radial = ∇apicobasal × ∇transmural
	
			transmural  = function_value(cv, qp, coordinate_system.transmural[dof_indices])
	
			# linear interpolation of rotation angle
			θ = (1-transmural) * endo_angle + (transmural) * epi_angle
			ϕ = (1-transmural) * endo_transversal_angle + (transmural) * epi_transversal_angle
	
			# Rodriguez rotation of ∇radial around ∇transmural with angle θ
			v = ∇radial / norm(∇radial)
			sinθ = sin(deg2rad(θ))
			cosθ = cos(deg2rad(θ))
			k = ∇transmural / norm(∇transmural)
			vᵣ = v * cosθ + (k × v) * sinθ + k * (k ⋅ v) * (1-cosθ)
			vᵣ = vᵣ / norm(vᵣ)

			# Rodriguez rotation of vᵣ around ∇radial with angle ϕ
			v = vᵣ / norm(vᵣ)
			sinϕ = sin(deg2rad(ϕ))
			cosϕ = cos(deg2rad(ϕ))
			k = ∇radial / norm(∇radial)
			vᵣ = v * cosϕ + (k × v) * sinϕ + k * (k ⋅ v) * (1-cosϕ)
			vᵣ = vᵣ / norm(vᵣ)

			elementwise_data[cellindex, qp] = vᵣ / norm(vᵣ)
		end
	end
			
	FieldCoefficient(elementwise_data, ip_fiber)
end

# ╔═╡ 97bc4a8f-1377-48cd-9d98-a28b1d464e8c
fiber_model_new = create_simple_fiber_model(coordinate_system, Lagrange{3, RefTetrahedron, 1}())

# ╔═╡ 620c34e6-48b0-49cf-8b3f-818107d0bc94
md"""
## Driver Code
"""

# ╔═╡ 9daf7714-d24a-4805-8c6a-62ceed0a4c9f
struct GMK2014
	passive_spring
	active_spring
end

# ╔═╡ c139e52f-05b6-4cfe-94fc-346535cba204
struct ContractileLinearSpring
	η
end

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

# ╔═╡ 374329cc-dcd5-407b-a4f2-83f21120577f
#λᵃ(Caᵢ) = (cos(pi*x)*(1-λᵃₘₐₓ) + 1.0)/2.0 + λᵃₘₐₓ/2.0
λᵃ(Caᵢ, β = 3.0, λᵃₘₐₓ = 0.7) = 1.0/(1+(0.5+atan(β*log(max(Caᵢ,1e-10)))/π))#*λᵃₘₐₓ

# ╔═╡ e4bf0f50-5028-4d09-8358-3615c7b2825c
function Ψᵃ(F, f₀, Caᵢ, mp::ContractileLinearSpring)
	@unpack η = mp
	
	M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
	Fᵃ = Tensors.unsafe_symmetric(one(F) + (λᵃ(Caᵢ) - 1.0) * M)
	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
	M̃ = f̃ ⊗ f̃

	FMF = Tensors.unsafe_symmetric(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))
	Iᵉ₄ = tr(FMF)
	return η / 2 * (Iᵉ₄ - 1)^2
end

# ╔═╡ bf061b21-1c0c-4673-8277-5faafff90851
plot(λᵃ, 0.0, 1.0)

# ╔═╡ 53e2e8ac-2110-4dfb-9cc4-824427a9ebf0
struct Passive2017Energy
	a
	a₁
	a₂
	b
	α₁
	α₂
	β
	η
end

# ╔═╡ f3289e09-4ca7-4e71-b798-32908ae23ce0
function Ψ(F, f₀, Caᵢ, mp::Passive2017Energy)
	# Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
    @unpack a, a₁, a₂, b, α₁, α₂, β, η = mp
	C = tdot(F)
    I₁ = tr(C)
	I₃ = det(C)
	I₄ = tr(C ⋅ f₀ ⊗ f₀)

	I₃ᵇ = I₃^b
	U = β * (I₃ᵇ + 1/I₃ᵇ - 2)^a
	Ψᵖ = α₁*(I₁/cbrt(I₃) - 3)^a₁ + α₂*max(I₄ - 1, 0.0)^a₂ + U

	M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
	Fᵃ = Tensors.unsafe_symmetric(one(F) + (λᵃ(Caᵢ) - 1.0) * M)
	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
	M̃ = f̃ ⊗ f̃

	Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀)
	FMF = Tensors.unsafe_symmetric(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))
	Iᵉ₄ = tr(FMF)
	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2

    return Ψᵖ + Ψᵃ
end

# ╔═╡ 2f59a097-e566-46c3-98a2-bedd35663073
struct BioNeoHooekan
	α
	β
	a
	b
	η
end

# ╔═╡ e7378847-977a-4282-841a-d568399d04cc
function Ψ(F, f₀, Caᵢ, mp::BioNeoHooekan)
	# Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
    @unpack a, b, α, β, η = mp
	C = tdot(F)
    I₁ = tr(C)
	I₃ = det(C)
	I₄ = f₀ ⋅ C ⋅ f₀

	I₃ᵇ = I₃^b
	U = β * (I₃ᵇ + 1/I₃ᵇ - 2)^a
	Ψᵖ = α*(I₁/cbrt(I₃) - 3) + U

	M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
	#Fᵃ = Tensors.unsafe_symmetric(one(F) + (λᵃ(Caᵢ) - 1.0) * M)
	Fᵃ = Tensors.unsafe_symmetric(λᵃ(Caᵢ) * M + (1.0/sqrt(λᵃ(Caᵢ)))*(one(F) - M))

	#Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀
	Fᵉ = F⋅inv(Fᵃ)
	Cᵉ = tdot(Fᵉ)
	I₃ᵉ = det(Cᵉ)
	I₃ᵉᵇ = I₃ᵉ^b
	Uᵃ = 1.0 * (I₃ᵉᵇ + 1/I₃ᵉᵇ - 2)^a

	Iᵉ₁ = tr(Cᵉ)
	Iᵉ₄ = f₀ ⋅ Cᵉ ⋅ f₀
	#Ψᵃ = η / 2 * (Iᵉ₄/cbrt(I₃ᵉ) - 1)^2 + 1.0*(Iᵉ₁/cbrt(I₃ᵉ)-3) + Uᵃ 
	#Ψᵃ = 1.00*(Iᵉ₁/cbrt(I₃ᵉ)-3) #+ Uᵃ 
	#Ψᵃ = η / 2 * (Iᵉ₄/cbrt(I₃ᵉ) - 1)^2 + Uᵃ
	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2 

    return Ψᵖ + Ψᵃ
end

# ╔═╡ 9d9da356-ac29-4a99-9a1a-e8f61141a8d1
struct ActiveNeoHookean
    μ::Float64
    λ::Float64
	η::Float64
end

# ╔═╡ 03dbf71b-c69a-4049-ad2f-1f78ae754fde
function Ψ(F, f₀, Caᵢ, mp::ActiveNeoHookean)
	@unpack μ, λ, η = mp
    J = det(F)
	C = tdot(F)
    I₁ = tr(C)

	Ψᵖ = μ / 2 * (I₁ - 3) - μ * log(J) + λ / 2 * log(J)^2

	M = f₀ ⊗ f₀
	Fᵃ = one(F) + (λᵃ(Caᵢ) - 1.0) * M
	#f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
	#M̃ = f̃ ⊗ f̃

	Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀)
	#Fᵉ = F⋅inv(Fᵃ)
	#FMF = Tensors.unsafe_symmetric(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))
	#FMF = Tensors.unsafe_symmetric(transpose(Fᵉ) ⋅ M̃ ⋅ Fᵉ)
	#Iᵉ₄ = tr(FMF)
	Cᵉ = tdot(Fᵉ)
	Iᵉ₁ = tr(Cᵉ)
	Iᵉ₄ = f₀ ⋅ Cᵉ ⋅ f₀
	#Iᵉ₄ = f̃ ⋅ tdot(Fᵉ) ⋅ f̃
	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2 #+ 1.0*(Iᵉ₁ - 3)

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
function constitutive_driver(F, f₀, Caᵢ, mp)
    # Compute all derivatives in one function call
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> Ψ(y, f₀, Caᵢ, mp), F, :all)

	# η = mp.η
	# M = f₀ ⊗ f₀
	# Fᵃ = one(F) + (λᵃ(Caᵢ) - 1.0) * M
	# f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
	# M̃ = f̃ ⊗ f̃

	# Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀)
	# Iᵉ₄ = tr(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))

	# ∂Ψ∂F += η * (Iᵉ₄ - 1.0) * Fᵉ ⋅ (M̃ + transpose(M̃)) ⋅ ((I - (1 - 1.0/λᵃ(Caᵢ))) * (f₀ ⊗ f₀))

	# ∂²Ψ∂F² += η * (Iᵉ₄ - 1.0) * Fᵉ ⋅ one(Tensor{4,3}) ⋅ (M̃ + transpose(M̃)) ⋅ ((I - (1 - 1.0/λᵃ(Caᵢ))) * (f₀ ⊗ f₀))
	# ∂²Ψ∂F² += η * Fᵉ ⋅ (M̃ + transpose(M̃)) ⋅ ((I - (1 - 1.0/λᵃ(Caᵢ))) * (f₀ ⊗ f₀)) ⊗ Fᵉ ⋅ (M̃ + transpose(M̃)) ⋅ ((I - (1 - 1.0/λᵃ(Caᵢ))) * (f₀ ⊗ f₀))

    return ∂Ψ∂F, ∂²Ψ∂F²
end;

# ╔═╡ b2b670d9-2fd7-4031-96bb-167db12475c7
function assemble_element!(cellid, Kₑ, residualₑ, cell, cv, fv, mp, uₑ, uₑ_prev, fiber_model, time)
	# TODO factor out
	kₛ = 0.001 # "Spring stiffness"
	kᵇ = 5.0 # Basal bending penalty

	Caᵢ(cellid,x,t) = t < 1.0 ? t : 2.0-t
	p = 7.5*(1.0/λᵃ(Caᵢ(0,0,time)) - 1.0/λᵃ(Caᵢ(0,0,0)))
	
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
		P, ∂P∂F = constitutive_driver(F, f₀(fiber_model, cellid, x_ref), Caᵢ(cellid, x_ref, time), mp)

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
		if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Epicardium")
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

		if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Base")
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
		if (cell.current_cellid.x, local_face_index) ∈ getfaceset(cell.grid, "Endocardium")
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
end;

# ╔═╡ aa57fc70-16f4-4013-a71e-a4099f0e5bd4
function assemble_global!(K, f, dh, cv, fv, mp, uₜ, uₜ₋₁, fiber_model, t)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    #@timeit "assemble" for cell in CellIterator(dh)
	for (cellid,cell) in enumerate(CellIterator(dh))
        global_dofs = celldofs(cell)
        uₑ = uₜ[global_dofs] # element dofs
        uₑ_prev = uₜ₋₁[global_dofs] # element dofs
		assemble_element!(cellid, ke, ge, cell, cv, fv, mp, uₑ, uₑ_prev, fiber_model, t)
        assemble!(assembler, global_dofs, ge, ke)
    end
end;

# ╔═╡ edb7ba53-ba21-4001-935f-ec9e0d7531be
function solve(grid, fiber_model)
	pvd = paraview_collection("GMK2014_LV.pvd");

	T = 2.0
	Δt = 0.1

    # Material parameters
    E = 2.0
    ν = 0.45
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
	# μ = 1.001
	# λ = 1.001
	η = 10.00
    mp = ActiveNeoHookean(μ, λ, η)
	# mp = Passive2017Energy(1.0, 2.6, 2.82, 2.0, 30.48, 7.25, 1.0, 10.0)
	# mp = BioNeoHooekan(1.0, 1.0, 1, 2, 10.0)

    # Finite element base
    ip = Lagrange{3, RefTetrahedron, 1}()
    ip_geo = Lagrange{3, RefTetrahedron, 1}()
    qr = QuadratureRule{3, RefTetrahedron}(2)
    qr_face = QuadratureRule{2, RefTetrahedron}(2)
    cv = CellVectorValues(qr, ip, ip_geo)
    fv = FaceVectorValues(qr_face, ip, ip_geo)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    dbcs = ConstraintHandler(dh)
    # Clamp base for now
    # dbc = Dirichlet(:u, getfaceset(grid, "Base"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
	# add!(dbcs, dbc)
    # dbc = Dirichlet(:u, getnodeset(grid, "Apex"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    # add!(dbcs, dbc)
    close!(dbcs)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)

	uₜ   = zeros(_ndofs)
	uₜ₋₁ = zeros(_ndofs)
    Δu   = zeros(_ndofs)

	#ref_vol = calculate_volume_deformed_mesh(uₜ,dh,cv);
	#min_vol = ref_vol
	#max_vol = ref_vol

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)

    NEWTON_TOL = 1e-8
	MAX_NEWTON_ITER = 100

	for t ∈ 0.0:Δt:T
		@info "t = " t

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
			@info "||g|| = " normg
	
	        if normg < NEWTON_TOL
	            break
	        elseif newton_itr > MAX_NEWTON_ITER
	            error("Reached maximum Newton iterations. Aborting.")
	        end
	
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
		fdata = zero(Vector{Vec{3}}(undef, getncells(grid)))

		for (cellid,cell) in enumerate(CellIterator(dh))
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
			fdata_cell = Vec{3}((0.0, 0.0, 0.0))

			nqp = getnquadpoints(cv)
			for qp in 1:nqp
		        dΩ = getdetJdV(cv, qp)

		        # Compute deformation gradient F
		        ∇u = function_gradient(cv, qp, uₑ)
		        F = one(∇u) + ∇u

				C = tdot(F)
				E = (C-one(C))/2.0
				x_ref = cv.qr.points[qp]
				fiber_direction = f₀(fiber_model, cellid, x_ref)
				fiber_direction /= norm(fiber_direction)

				E_ff_cell += fiber_direction ⋅ E ⋅ fiber_direction
				
				fiber_direction_current = F⋅fiber_direction
				fiber_direction_current /= norm(fiber_direction_current)

				E_ff_cell2 += fiber_direction_current ⋅ E ⋅ fiber_direction_current

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

				frefdata_cell += fiber_direction

				fdata_cell += fiber_direction_current
			end

			E_ff[cellid] = E_ff_cell / nqp
			E_ff2[cellid] = E_ff_cell2 / nqp
			E_cc[cellid] = E_cc_cell / nqp
			E_rr[cellid] = E_rr_cell / nqp
			E_ll[cellid] = E_ll_cell / nqp
			Jdata[cellid] = Jdata_cell / nqp
			frefdata[cellid] = frefdata_cell / nqp
			fdata[cellid] = fdata_cell / nqp
		end

	    # Save the solution
		vtk_grid("GMK2014-LV-$t.vtu", dh) do vtk
            vtk_point_data(vtk,dh,uₜ)
	        vtk_cell_data(vtk,hcat(frefdata...),"Reference Fiber Data")
			vtk_cell_data(vtk,hcat(fdata...),"Current Fiber Data")
	        vtk_cell_data(vtk,E_ff,"E_ff")
	        vtk_cell_data(vtk,E_ff2,"E_ff2")
	        vtk_cell_data(vtk,E_cc,"E_cc")
	        vtk_cell_data(vtk,E_rr,"E_rr")
	        vtk_cell_data(vtk,E_ll,"E_ll")
	        vtk_cell_data(vtk,Jdata,"J")
            vtk_save(vtk)
	        pvd[t] = vtk
	    end

		#min_vol = min(min_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
		#max_vol = max(max_vol, calculate_volume_deformed_mesh(uₜ,dh,cv));
	end

	#println("Compression: ", (ref_vol/min_vol - 1.0)*100, "%")
	#println("Expansion: ", (ref_vol/max_vol - 1.0)*100, "%")
	
	vtk_save(pvd);

	return uₜ
end

# ╔═╡ c24c7c84-9953-4886-9b34-70bdf942fe1b
function calculate_element_volume(cell, cellvalues_u, uₑ)
    reinit!(cellvalues_u, cell)
    evol::Float64=0.0;
    @inbounds for qp in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        ∇u = function_gradient(cellvalues_u, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        evol += J * dΩ
    end
    return evol
end;

# ╔═╡ bb150ecb-f844-48d1-9a09-47abfe6db89c
function calculate_volume_deformed_mesh(w, dh::DofHandler, cellvalues_u)
    evol::Float64 = 0.0;
    @inbounds for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        nu = getnbasefunctions(cellvalues_u)
        global_dofs_u = global_dofs[1:nu]
        uₑ = w[global_dofs_u]
        δevol = calculate_element_volume(cell, cellvalues_u, uₑ)
        evol += δevol;
    end
    return evol
end;

# ╔═╡ 8cfeddaa-c67f-4de8-b81c-4fbb7e052c50
#solve(grid, fiber_model_new)

# ╔═╡ 7b9a3e74-dd0f-497b-aec2-73b5b9242b12
function solve_test()
	# geo = Tetrahedron
	# refgeo = RefTetrahedron
	# intorder = 1

	geo = Hexahedron
	refgeo = RefCube
	intorder = 2

	N = 3
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(geo, (N, N, N), left, right)
	addfaceset!(grid, "Endocardium", Set{Ferrite.FaceIndex}([]))
	addfaceset!(grid, "Base", Set{Ferrite.FaceIndex}([]))
	addfaceset!(grid, "Epicardium", Set{Ferrite.FaceIndex}([]))

	f₀data = Vector{Vec{3}}(undef, getncells(grid))
	for cellindex ∈ 1:getncells(grid)
		f₀data[cellindex] = Vec{3}((1.0, 0.0, 0.0))
	end
	fiber_model = PiecewiseConstantFiberModel(f₀data, [], [])

	pvd = paraview_collection("GMK2014_cube.pvd");

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
	mp = Passive2017Energy(1.0, 2.6, 2.82, 2.0, 30.48, 7.25, 1.0, 100.0)
	#mp = BioNeoHooekan(4.0, 10.25, 1, 2, 10.0)

    # Finite element base
    ip = Lagrange{3, refgeo, 1}()
    ip_geo = Lagrange{3, refgeo, 1}()
    qr = QuadratureRule{3, refgeo}(intorder)
    qr_face = QuadratureRule{2, refgeo}(intorder)
    cv = CellVectorValues(qr, ip, ip_geo)
    fv = FaceVectorValues(qr_face, ip, ip_geo)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    dbcs = ConstraintHandler(dh)
    # Clamp three sides
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> [0.0], [1])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "front"), (x,t) -> [0.0], [2])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "bottom"), (x,t) -> [0.0], [3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
	
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
		@info "t = " t

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
			@info "||g|| = " normg
	
	        if normg < NEWTON_TOL
	            break
	        elseif newton_itr > MAX_NEWTON_ITER
	            error("Reached maximum Newton iterations. Aborting.")
	        end

			@info det(K)
			Δu = K \ g
	
	        apply_zero!(Δu, dbcs)

			uₜ .-= Δu # Current guess
	    end

		# Compute some elementwise measures
		E_ff = zeros(getncells(grid))
		E_cc = zeros(getncells(grid))
		E_ll = zeros(getncells(grid))
		E_rr = zeros(getncells(grid))
		Jdata = zeros(getncells(grid))
		fdata = copy(fiber_model.f₀data)
		for (cellid,cell) in enumerate(CellIterator(dh))
			reinit!(cv, cell)
			global_dofs = celldofs(cell)
        	uₑ = uₜ[global_dofs] # element dofs
			
			E_ff_cell = 0.0
			E_cc_cell = 0.0
			E_rr_cell = 0.0
			E_ll_cell = 0.0

			nqp = getnquadpoints(cv)
			for qp in 1:nqp
		        dΩ = getdetJdV(cv, qp)

		        # Compute deformation gradient F
		        ∇u = function_gradient(cv, qp, uₑ)
		        F = one(∇u) + ∇u

				C = tdot(F)
				E = (C-one(C))/2.0
				x_ref = qr.points[qp]
				#fiber_direction = F ⋅ f₀(fiber_model, cellid, x_ref)
				fiber_direction = f₀(fiber_model, cellid, x_ref)
				fiber_direction /= norm(fiber_direction)
				
				E_ff_cell += fiber_direction ⋅ E ⋅ fiber_direction
				E_ff_cell += fiber_direction ⋅ E ⋅ fiber_direction

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
		
				Jdata[cellid] = det(F)

				fdata[cellid] = F⋅fiber_model.f₀data[cellid]
			end

			E_ff[cellid] = E_ff_cell / nqp
			E_cc[cellid] = E_cc_cell / nqp
			E_rr[cellid] = E_rr_cell / nqp
			E_ll[cellid] = E_ll_cell / nqp
		end
	
	    # Save the solution
		vtk_grid("GMK2014-cube-$t.vtu", dh) do vtk
            vtk_point_data(vtk,dh,uₜ)
	        vtk_cell_data(vtk,hcat(fiber_model.f₀data...),"Reference Fiber Data")
			vtk_cell_data(vtk,hcat(fdata...),"Current Fiber Data")
	        vtk_cell_data(vtk,E_ff,"E_ff")
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

# ╔═╡ 76b6b377-287a-45a0-b02b-92e205dde287
#solve_test();

# ╔═╡ 4b76b944-2054-47ed-bbcb-b1412643fed0
md"""
What goes wrong: The ventricle elongates in longitudinal direction.

Hotfix: Playing around with fiber and sheet angle.

What it seems to be not
1. Sheetlet angle
2. $I_4$ in passive portion
3. incompressibility
4. choice of boundary conditions (we can partially force a contraction in longitudinal direction, but the cavity surface is wrong then)

I think we are missing some energy pirtion, either $I_{4s}$ or $I_{8fs}$.
"""

# ╔═╡ df771870-34a6-45c4-a5c9-0c4fccf78941
function compute_midmyocardial_section_coordinate_system(grid)
	ip = Lagrange{3, RefCube, 1}()
	qr = QuadratureRule{3, RefCube}(2)
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

	vtk_grid("coordinates_apicobasal", dh) do vtk
	    vtk_point_data(vtk, dh, apicobasal)
	end

	return LVCoordinateSystem(dh, qr, cellvalues, transmural, apicobasal)
end

# ╔═╡ 3aa86aa9-308a-494f-bb8c-0983f22dfa07
function generate_ring_mesh(ne_c, ne_r, ne_l; radial_inner::T = Float64(0.75), radial_outer::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = ne_c*ne_r*ne_l;
    n_nodes_c = ne_c; n_nodes_r = ne_r+1; n_nodes_l = ne_l+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    coords_c = range(0.0, stop=2*π, length=n_nodes_c+1)
    coords_r = range(radial_inner, stop=radial_outer, length=n_nodes_r)
    coords_l = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = coords_r[j]-0.03*coords_l[k]/maximum(abs.(coords_l))
        push!(nodes, Node((radius*cos(coords_c[i]), radius*sin(coords_c[i]), coords_l[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k in 1:ne_l, j in 1:ne_r, i in 1:ne_c
        i_next = (i == ne_c) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:ne_tot),(ne_c, ne_r, ne_l))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FaceIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            #[FaceIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FaceIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            #[FaceIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FaceIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["Myocardium"] = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facesets["Endocardium"]  = Set{FaceIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facesets["Epicardium"]   = Set{FaceIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facesets["Base"]    = Set{FaceIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    return Grid(cells, nodes, facesets=facesets)
end

# ╔═╡ d387f754-15cc-4875-8130-ca7482faac94
function solve_test_ring()
	# geo = Tetrahedron
	# refgeo = RefTetrahedron
	# intorder = 1

	geo = Hexahedron
	refgeo = RefCube
	order = 1
	intorder = 2

    grid = generate_ring_mesh(40,6,4)
	coordinate_system = compute_midmyocardial_section_coordinate_system(grid)
	#fiber_model = create_simple_fiber_model(coordinate_system, Lagrange{3, refgeo, 1}(), endo_angle = 50.0, epi_angle = -70.0)
	# α from "Transmural left ventricular mechanics underlying torsional recoil during relaxation."
	fiber_model = create_simple_fiber_model(coordinate_system, Lagrange{3, refgeo, 1}(), endo_angle = 80.0, epi_angle = -55.0)
	#fiber_model = create_simple_fiber_model(coordinate_system, Lagrange{3, refgeo, 1}(), endo_angle = 40.0, epi_angle = -60.0, endo_transversal_angle = 5.0, epi_transversal_angle = 1.0)

	pvd = paraview_collection("GMK2014_ring.pvd");

	T = 1.0
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
	mp = BioNeoHooekan(0.25, 4.00, 1, 2, 15.0)

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

	vtk_save(vtk_grid("GMK2014-ring.vtu", dh))

    dbcs = ConstraintHandler(dh)
    # # Clamp three sides
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
		fdata = zero(Vector{Vec{3}}(undef, getncells(grid)))

		for (cellid,cell) in enumerate(CellIterator(dh))
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
			fdata_cell = Vec{3}((0.0, 0.0, 0.0))

			nqp = getnquadpoints(cv)
			for qp in 1:nqp
		        dΩ = getdetJdV(cv, qp)

		        # Compute deformation gradient F
		        ∇u = function_gradient(cv, qp, uₑ)
		        F = one(∇u) + ∇u

				C = tdot(F)
				E = (C-one(C))/2.0
				x_ref = cv.qr.points[qp]
				fiber_direction = f₀(fiber_model, cellid, x_ref)
				fiber_direction /= norm(fiber_direction)

				E_ff_cell += fiber_direction ⋅ E ⋅ fiber_direction
				
				fiber_direction_current = F⋅fiber_direction
				fiber_direction_current /= norm(fiber_direction_current)

				E_ff_cell2 += fiber_direction_current ⋅ E ⋅ fiber_direction_current

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

				frefdata_cell += fiber_direction

				fdata_cell += fiber_direction_current
			end

			E_ff[cellid] = E_ff_cell / nqp
			E_ff2[cellid] = E_ff_cell2 / nqp
			E_cc[cellid] = E_cc_cell / nqp
			E_rr[cellid] = E_rr_cell / nqp
			E_ll[cellid] = E_ll_cell / nqp
			Jdata[cellid] = Jdata_cell / nqp
			frefdata[cellid] = frefdata_cell / nqp
			fdata[cellid] = fdata_cell / nqp
		end

	    # Save the solution
		vtk_grid("GMK2014-ring-$t.vtu", dh) do vtk
            vtk_point_data(vtk,dh,uₜ)
	        vtk_cell_data(vtk,hcat(frefdata...),"Reference Fiber Data")
			vtk_cell_data(vtk,hcat(fdata...),"Current Fiber Data")
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

# ╔═╡ 7fdacae4-747a-46fc-a0ca-cb6ea5c47bc8
solve_test_ring()

# ╔═╡ Cell order:
# ╟─6e43f86d-6341-445f-be8d-146eb0447457
# ╠═48685583-8831-4797-bde9-f2032ef42837
# ╟─f6b5bde0-bcdc-41de-b64e-29b9e60b06c0
# ╟─94fc407e-d72e-4b44-9d30-d66c433b954a
# ╟─a81fc16e-c280-4b38-8c6a-18714db7840c
# ╠═d314acc8-22fc-40f8-bcc6-729b6e208f70
# ╠═bc3bbb45-c263-4f29-aa6f-647e8dbf3466
# ╟─f4a00f5d-f042-4804-9be1-d24d5046fd0a
# ╠═610c857e-a699-48f6-b18b-df8337287122
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
# ╠═aaf500d1-c0ef-4839-962a-139503e6fcc0
# ╠═dedc939c-b789-4a8e-bc5e-e70e40bb6b8d
# ╠═0c863dcc-c5aa-47a7-9941-97b7a392febd
# ╠═cb09e0f9-fd04-4cb9-bed3-a1e97cb6d478
# ╠═97bc4a8f-1377-48cd-9d98-a28b1d464e8c
# ╟─620c34e6-48b0-49cf-8b3f-818107d0bc94
# ╠═9daf7714-d24a-4805-8c6a-62ceed0a4c9f
# ╠═c139e52f-05b6-4cfe-94fc-346535cba204
# ╠═e4bf0f50-5028-4d09-8358-3615c7b2825c
# ╠═2eae0e78-aba6-49f5-84b1-dff9c087b391
# ╠═0658ec98-5883-4859-93dc-f9e9f52f9f63
# ╠═374329cc-dcd5-407b-a4f2-83f21120577f
# ╠═bf061b21-1c0c-4673-8277-5faafff90851
# ╟─4ff78cdf-1efc-4c00-91a3-4c29f3d27305
# ╠═53e2e8ac-2110-4dfb-9cc4-824427a9ebf0
# ╠═f3289e09-4ca7-4e71-b798-32908ae23ce0
# ╠═2f59a097-e566-46c3-98a2-bedd35663073
# ╠═e7378847-977a-4282-841a-d568399d04cc
# ╠═9d9da356-ac29-4a99-9a1a-e8f61141a8d1
# ╠═03dbf71b-c69a-4049-ad2f-1f78ae754fde
# ╠═641ad832-1b22-44d0-84f0-bfe15ecd6246
# ╠═b2b670d9-2fd7-4031-96bb-167db12475c7
# ╠═aa57fc70-16f4-4013-a71e-a4099f0e5bd4
# ╠═edb7ba53-ba21-4001-935f-ec9e0d7531be
# ╠═c24c7c84-9953-4886-9b34-70bdf942fe1b
# ╠═bb150ecb-f844-48d1-9a09-47abfe6db89c
# ╠═8cfeddaa-c67f-4de8-b81c-4fbb7e052c50
# ╠═7b9a3e74-dd0f-497b-aec2-73b5b9242b12
# ╠═76b6b377-287a-45a0-b02b-92e205dde287
# ╟─4b76b944-2054-47ed-bbcb-b1412643fed0
# ╠═d387f754-15cc-4875-8130-ca7482faac94
# ╠═7fdacae4-747a-46fc-a0ca-cb6ea5c47bc8
# ╠═df771870-34a6-45c4-a5c9-0c4fccf78941
# ╠═3aa86aa9-308a-494f-bb8c-0983f22dfa07
