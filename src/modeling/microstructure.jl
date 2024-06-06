# TODO Merge the two coefficient below and alias the name
struct AnisotropicPlanarMicrostructureModel{FiberCoefficientType, SheetletCoefficientType}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
end

function evaluate_coefficient(fsn::AnisotropicPlanarMicrostructureModel, cell_cache, qp::QuadraturePoint{2}, t)
    f = evaluate_coefficient(fsn.fiber_coefficient, cell_cache, qp, t)
    s = evaluate_coefficient(fsn.sheetlet_coefficient, cell_cache, qp, t)

    return SVector((f, s))
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

    return SVector((f, s, n))
end

"""
    streeter_type_fsn(transmural_direction::Vec{3}, circumferential_direction::Vec{3}, apicobasal_direction::Vec{3}, helix_angle, transversal_angle, sheetlet_pseudo_angle, make_orthogonal=true)

Compute fiber, sheetlet and normal direction from the transmural, circumferential, apicobasal directions
in addition to given helix, transversal and sheetlet angles. The theory is based on the classical work by
[StreSpoPatRosSon:1969:foc](@citet).
"""
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
    create_simple_microstructure_model(coordinate_system, ip_component::VectorInterpolationCollection; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_angle = 0.0, make_orthogonal=true)

Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.

!!! note
    FIXME! Sheetlet angle construction is broken (i.e. does not preserve input angle).
"""
function create_simple_microstructure_model(coordinate_system, ip_collection::VectorizedInterpolationCollection{3}; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_pseudo_angle = 0.0, make_orthogonal=true)
    @unpack dh = coordinate_system

    check_subdomains(dh)

    first_cell = getcells(Ferrite.get_grid(dh), 1)
    ip = getinterpolation(ip_collection, first_cell)
    n_basefuns = getnbasefunctions(ip.ip)

    elementwise_data_f = zero(Array{Vec{3,Float64}, 2}(undef, n_basefuns, getncells(dh.grid)))
    elementwise_data_s = zero(Array{Vec{3,Float64}, 2}(undef, n_basefuns, getncells(dh.grid)))
    elementwise_data_n = zero(Array{Vec{3,Float64}, 2}(undef, n_basefuns, getncells(dh.grid)))

    cv_collection = CellValueCollection(NodalQuadratureRuleCollection(ip_collection.base), ip_collection.base)
    cv = getcellvalues(cv_collection, first_cell)

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
            elementwise_data_f[qp.i, cellindex] = f₀
            elementwise_data_s[qp.i, cellindex] = s₀
            elementwise_data_n[qp.i, cellindex] = n₀
        end
    end

    OrthotropicMicrostructureModel(
        FieldCoefficient(elementwise_data_f, ip_collection),
        FieldCoefficient(elementwise_data_s, ip_collection),
        FieldCoefficient(elementwise_data_n, ip_collection)
    )
end
