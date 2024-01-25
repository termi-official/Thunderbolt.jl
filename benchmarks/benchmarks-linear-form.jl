using BenchmarkTools, Thunderbolt, StaticArrays

grid = generate_grid(celltype, (1,1,1))
cell_cache = Ferrite.CellCache(grid)
reinit!(cell_cache,1)

ip_collection = LagrangeCollection{1}()
ip = getinterpolation(ip_collection, grid.cells[1])
qr_collection = QuadratureRuleCollection(2)
qr = getquadraturerule(qr_collection, grid.cells[1])
cv = CellValues(qr, ip)
ac = AnalyticalCoefficient(
    (x,t) -> norm(x)+t,
    CoordinateSystemCoefficient(
        CartesianCoordinateSystem(grid)
    )
)
b = zeros(8)
element_cache = Thunderbolt.AnalyticalCoefficientElementCache(ac, [SVector((0.0,1.0))], cv)
@btime Thunderbolt.assemble_element!($b, $cell_cache, $element_cache, 0.0)
