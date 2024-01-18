# TODO Merge the two coefficient below and alias the name
struct AnisotropicPlanarMicrostructureModel{FiberCoefficientType, SheetletCoefficientType}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
end

function evaluate_coefficient(fsn::AnisotropicPlanarMicrostructureModel, cell_cache, qp::QuadraturePoint{2}, t)
    f = evaluate_coefficient(fsn.fiber_coefficient, cell_cache, qp, t)
    s = evaluate_coefficient(fsn.sheetlet_coefficient, cell_cache, qp, t)

    f, s
end

struct OrthotropicMicrostructureModel{FiberCoefficientType, SheetletCoefficientType, NormalCoefficientType}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
    normal_coefficient::NormalCoefficientType
end

function evaluate_coefficient(fsn::OrthotropicMicrostructureModel, cell_cache, qp::QuadraturePoint{3}, t)
    f = evaluate_coefficient(fsn.fiber_coefficient, cell_cache, qp, t)
    s = evaluate_coefficient(fsn.sheetlet_coefficient, cell_cache, qp, t)
    n = evaluate_coefficient(fsn.normal_coefficient, cell_cache, qp, t)

    f, s, n
end

# TODO: citation
function streeter_type_fsn(transmural_direction, circumferential_direction, apicobasal_direction, helix_angle, transversal_angle, sheetlet_pseudo_angle, make_orthogonal=true)
    # First we construct the helix rotation ...
    f₀ = rotate_around(circumferential_direction, transmural_direction, helix_angle)
    f₀ /= norm(f₀)
    # ... followed by the transversal_angle ...
    f₀ = rotate_around(f₀, apicobasal_direction, transversal_angle)
    f₀ /= norm(f₀)

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

    # Compute normal :)
    n₀ = f₀ × s₀
    n₀ /= norm(n₀)

    return f₀, s₀, n₀
end

"""
    create_simple_microstructure_model(coordinate_system, ip_component::Interpolation{ref_shape}, ip_geo; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_angle = 0.0) where {dim}

Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.

!!! note Sheetlet angle construction is broken (i.e. does not preserve input angle). FIXME!
"""
function create_simple_microstructure_model(coordinate_system, ip_collection::VectorInterpolationCollection, ip_geo_collection::VectorizedInterpolationCollection; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_pseudo_angle = 0.0, make_orthogonal=true)
    @unpack dh = coordinate_system

    ref_shape = getrefshape(getcells(Ferrite.get_grid(dh), 1))
    ip = getinterpolation(ip_collection, ref_shape)
    ip_geo = getinterpolation(ip_geo_collection, ref_shape)
    n_basefuns = getnbasefunctions(ip.ip)

    elementwise_data_f = zero(Array{Vec{3,Float64}, 2}(undef, getncells(dh.grid), n_basefuns))
    elementwise_data_s = zero(Array{Vec{3,Float64}, 2}(undef, getncells(dh.grid), n_basefuns))
    elementwise_data_n = zero(Array{Vec{3,Float64}, 2}(undef, getncells(dh.grid), n_basefuns))

    qr_fiber = generate_nodal_quadrature_rule(ip.ip)
    cv = create_cellvalues(coordinate_system, qr_fiber, ip_geo)

    for (cellindex,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
        dof_indices = celldofs(cell)

        for qp in QuadratureIterator(cv)
            # TODO grab these via some interface!
            apicobasal_direction = function_gradient(cv, qp, coordinate_system.u_apicobasal[dof_indices])
            apicobasal_direction /= norm(apicobasal_direction)
            transmural_direction = function_gradient(cv, qp, coordinate_system.u_transmural[dof_indices])
            transmural_direction /= norm(transmural_direction)
            circumferential_direction = apicobasal_direction × transmural_direction
            circumferential_direction /= norm(circumferential_direction)

            transmural  = function_value(cv, qp, coordinate_system.u_transmural[dof_indices])

            # linear interpolation of rotation angle
            helix_angle       = (1-transmural) * endo_helix_angle + (transmural) * epi_helix_angle
            transversal_angle = (1-transmural) * endo_transversal_angle + (transmural) * epi_transversal_angle

            f₀, s₀, n₀ = streeter_type_fsn(transmural_direction, circumferential_direction, apicobasal_direction, helix_angle, transversal_angle, sheetlet_pseudo_angle, make_orthogonal)
            elementwise_data_f[cellindex, qp.i] = f₀
            elementwise_data_s[cellindex, qp.i] = s₀
            elementwise_data_n[cellindex, qp.i] = n₀
        end
    end

    OrthotropicMicrostructureModel(
        FieldCoefficient(elementwise_data_f, ip),
        FieldCoefficient(elementwise_data_s, ip),
        FieldCoefficient(elementwise_data_n, ip)
    )
end
