using Thunderbolt

using StaticArrays
using Test

grid = generate_grid(Quadrilateral, (2,2))
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,1}())
close!(dh)
qrc = QuadratureRuleCollection{2}()

cs = CartesianCoordinateSystem(grid)

propertynames(dh)



protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> 1.0, CoordinateSystemCoefficient(cs)),
                [SVector((0.0, 1.0))]
            )

 



linop = Thunderbolt.LinearOperator(
    zeros(ndofs(dh)),
    protocol,
    qrc,
    dh,
)

Thunderbolt.update_operator!(linop,0.0)
@test linop.b ≈ [0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.5, 0.25, 0.25]

# sdh = dh.subdofhandlers[1]
# field_name = first(dh.field_names)

# ip          = Ferrite.getfieldinterpolation(sdh, field_name)


# element_qr  = getquadraturerule(qrc, sdh)

# # Build evaluation caches
# element_cache = Thunderbolt.setup_element_cache(protocol, element_qr, ip, sdh)



cuda_op = Thunderbolt.init_linear_operator(Thunderbolt.BackendCUDA,protocol, qrc, dh)
Thunderbolt.update_operator!(cuda_op,0.0)


propertynames(cuda_op)
cuda_op.op.b


sdh = dh.subdofhandlers[1]
ip = Ferrite.getfieldinterpolation(sdh, :u)
element_qr  = getquadraturerule(qrc, sdh)
element_cache = Thunderbolt.setup_element_cache(protocol, element_qr, ip, sdh)


using CUDA
using Adapt


# function Adapt.adapt_structure(to, cv::StaticInterpolationValues)
#     return Adapt.adapt(to, cv)
# end

function coeff_kernel(sv)
    @cushow sv.Nξ[1,2]
    return nothing
end

sv = element_cache.cc.coordinate_system_cache
cv = element_cache.cv
sv1 = StaticInterpolationValues(cv.fun_values)
@cuda threads = 1  blocks = 1 coeff_kernel(sv)
element_cache

