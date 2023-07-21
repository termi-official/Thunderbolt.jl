"""
"""
struct FieldCoefficient{TA,IP<:Interpolation}
    elementwise_data::TA #3d array (element_idx, base_fun_idx, dim)
    ip::IP
end

"""
"""
function value(coeff::FieldCoefficient{TA,IP}, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim,TA,IP}
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
value(coeff::ConstantFieldCoefficient{T}, cell_id::Int, ξ::Vec{dim}, t::Float64=0.0) where {dim,T} = coeff.val



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
function generate_nodal_quadrature_rule(ip::Interpolation{ref_shape, order}) where {ref_shape, order}
    n_base = Ferrite.getnbasefunctions(ip)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape, Float64}(ones(length(positions)), positions)
end

"""
    create_simple_fiber_model(coordinate_system, ip_fiber::Interpolation{ref_shape}, ip_geo; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_angle = 0.0) where {dim}

Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.

!!! note Sheetlet angle construction is broken (i.e. does not preserve input angle). FIXME!
"""
function create_simple_fiber_model(coordinate_system, ip_fiber::Interpolation{ref_shape}, ip_geo; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_pseudo_angle = 0.0, make_orthogonal=true) where {dim, ref_shape <: AbstractRefShape{dim}}
    @unpack dh = coordinate_system

    n_basefuns = getnbasefunctions(ip_fiber)

    elementwise_data_f = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))
    elementwise_data_s = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))
    elementwise_data_n = zero(Array{Vec{dim}, 2}(undef, getncells(dh.grid), n_basefuns))

    qr_fiber = generate_nodal_quadrature_rule(ip_fiber)
    cv = create_cellvalues(coordinate_system, qr_fiber, ip_geo)

    for (cellindex,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
        dof_indices = celldofs(cell)

        for qp in 1:getnquadpoints(cv)
            # TODO grab these via some interface!
            apicobasal_direction = function_gradient(cv, qp, coordinate_system.u_apicobasal[dof_indices])
            apicobasal_direction /= norm(apicobasal_direction)
            transmural_direction = function_gradient(cv, qp, coordinate_system.u_transmural[dof_indices])
            transmural_direction /= norm(transmural_direction)
            circumferential_direction = apicobasal_direction × transmural_direction
            circumferential_direction /= norm(circumferential_direction)

            transmural  = function_value(cv, qp, coordinate_system.u_transmural[dof_indices])

            # linear interpolation of rotation angle
            # TODO pass functions or similar for these!
            helix_angle       = (1-transmural) * endo_helix_angle + (transmural) * epi_helix_angle
            transversal_angle = (1-transmural) * endo_transversal_angle + (transmural) * epi_transversal_angle

            # First we construct the helix rotation ...
            f₀ = rotate_around(circumferential_direction, transmural_direction, helix_angle)
            f₀ /= norm(f₀)
            # ... followed by the transversal_angle ...
            f₀ = rotate_around(f₀, apicobasal_direction, transversal_angle)
            f₀ /= norm(f₀)
            # ... and store it.
            elementwise_data_f[cellindex, qp] = f₀

            # Then we construct the the orthogonal sheetlet vector ...
            s₀ = rotate_around(circumferential_direction, transmural_direction, helix_angle+π/2.0)
            s₀ /= norm(f₀)
            # FIXME this does not preserve the sheetlet angle
            s₀ = unproject(s₀, -transmural_direction, sheetlet_pseudo_angle)
            if make_orthogonal
                s₀ = orthogonalize(s₀/norm(s₀), f₀)
            end
            # TODO replace above with an implementation of the following pseudocode
            # 1. Compute plane via P = I - f₀ ⊗ f₀
            # 2. Eigen decomposition E of P
            # 3. Compute a generalized eigenvector s' from the non-zero eigenvalue (with ||s'||=1) from E such that <f₀,s'> minimal
            # 4. Compute s₀ by rotating s' around f₀ such that cos(sheetlet angle) = <s',f₀>
            s₀ /= norm(s₀)
            elementwise_data_s[cellindex, qp] = s₀

            # Compute normal :)
            n₀ = f₀ × s₀
            elementwise_data_n[cellindex, qp] = n₀
            elementwise_data_n[cellindex, qp] /= norm(n₀)
        end
    end

    OrthotropicMicrostructureModel(
        FieldCoefficient(elementwise_data_f, ip_fiber),
        FieldCoefficient(elementwise_data_s, ip_fiber),
        FieldCoefficient(elementwise_data_n, ip_fiber)
    )     
end
