"""
"""
struct FieldCoefficient{dim}
	elementwise_data #3d array (element_idx, base_fun_idx, dim)
	ip::Interpolation{dim}
end

"""
"""
function value(coeff::FieldCoefficient{dim}, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim}
	@unpack elementwise_data, ip = coeff

    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(Vec{dim, Float64})

    @inbounds for i in 1:n_base_funcs
        val += Ferrite.value(ip, i, ξ) * elementwise_data[cell_id, i]
    end
    return val / norm(val)
end

"""
"""
struct ConstantFieldCoefficient{T}
    val::T
end

"""
"""
value(coeff::ConstantFieldCoefficient, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim} = coeff.val



struct OrthotropicMicrostructureModel{FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
    normal_coefficient::NormalCoefficientType
end

function directions(fsn::OrthotropicMicrostructureModel, cell_id::Int, ξ::Vec{dim}, t = 0.0) where {dim}
    f₀ = value(fsn.fiber_coefficient, cell_id, ξ, t)
    s₀ = value(fsn.sheetlet_coefficient, cell_id, ξ, t)
    n₀ = value(fsn.normal_coefficient, cell_id, ξ, t)

    f₀, s₀, n₀
end


"""
"""
function generate_nodal_quadrature_rule(ip::Interpolation{dim, ref_shape, order}) where {dim, ref_shape, order}
	n_base = Ferrite.getnbasefunctions(ip)
	positions = Ferrite.reference_coordinates(ip)
	return QuadratureRule{dim, ref_shape, Float64}(ones(length(positions)), positions)
end

"""
Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.
"""
function create_simple_fiber_model(coordinate_system, ip_fiber::Interpolation{dim}; endo_angle = 80.0, epi_angle = -65.0, endo_transversal_angle = 0.0, epi_transversal_angle = 0.0) where {dim}
	@unpack dh = coordinate_system

	ip = dh.field_interpolations[1] #TODO refactor this. Pls.

	n_basefuns = getnbasefunctions(ip_fiber)

	elementwise_data_f = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))
	elementwise_data_s = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))
    elementwise_data_n = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))

	qr_fiber = generate_nodal_quadrature_rule(ip_fiber)
	cv = CellScalarValues(qr_fiber, ip)

	for (cellindex,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
		dof_indices = celldofs(cell)

		for qp in 1:getnquadpoints(cv)
			# compute fiber direction
			∇apicobasal = function_gradient(cv, qp, coordinate_system.u_apicobasal[dof_indices])
			∇transmural = function_gradient(cv, qp, coordinate_system.u_transmural[dof_indices])
			∇radial = ∇apicobasal × ∇transmural

			transmural  = function_value(cv, qp, coordinate_system.u_transmural[dof_indices])

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

			elementwise_data_f[cellindex, qp] = vᵣ / norm(vᵣ)

			v = -∇apicobasal / norm(∇apicobasal)
			# v = -∇transmural / norm(∇transmural)
			sinϕ = sin(deg2rad(ϕ))
			cosϕ = cos(deg2rad(ϕ))
			k = ∇radial / norm(∇radial)
			# k = ∇apicobasal / norm(∇apicobasal)
			vᵣ = v * cosϕ + (k × v) * sinϕ + k * (k ⋅ v) * (1-cosϕ)
			vᵣ = vᵣ / norm(vᵣ)

			vᵣ = vᵣ - (elementwise_data_f[cellindex, qp]⋅vᵣ)*elementwise_data_f[cellindex, qp]
			elementwise_data_s[cellindex, qp] = vᵣ / norm(vᵣ)

            elementwise_data_n[cellindex, qp] = Tensors.cross(elementwise_data_f[cellindex, qp], elementwise_data_s[cellindex, qp])
            elementwise_data_n[cellindex, qp] /= norm(elementwise_data_n[cellindex, qp])
        end
	end

	OrthotropicMicrostructureModel(
        FieldCoefficient(elementwise_data_f, ip_fiber),
        FieldCoefficient(elementwise_data_s, ip_fiber),
        FieldCoefficient(elementwise_data_n, ip_fiber)
    )     
end
