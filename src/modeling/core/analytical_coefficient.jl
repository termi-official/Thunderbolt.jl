
"""
    AnalyticalCoefficient(f::Function, cs::CoordinateSystemCoefficient)

A coefficient given as the analytical function f(x,t) in the specified coordiante system.
"""
struct AnalyticalCoefficient{F<:Function, CSYS<:CoordinateSystemCoefficient}
    f::F
    coordinate_system_coefficient::CSYS
end

function evaluate_coefficient(coeff::F, cell_cache, qp::QuadraturePoint{<:Any,T}, t) where {F <: AnalyticalCoefficient, T}
    x = evaluate_coefficient(coeff.coordinate_system_coefficient, cell_cache, qp, t)
    return coeff.f(x, t)
end


"""
    AnalyticalCoefficientElementCache(f(x,t)->..., nonzero_in_time_intervals, cellvalues)

Analytical coefficient described by a function in space and time.
Can be sparse in time.
"""
struct AnalyticalCoefficientElementCache{F <: AnalyticalCoefficient, T, CV}
    f::F
    nonzero_intervals::Vector{SVector{2,T}}
    cv::CV
end
duplicate_for_parallel(ec::AnalyticalCoefficientElementCache) = AnalyticalCoefficientElementCache(ec.f, ec.nonzero_intervals, ec.cv)

@inline function assemble_element!(bₑ::AbstractVector, cell::CellCache, element_cache::AnalyticalCoefficientElementCache, time)
    _assemble_element!(bₑ, getcoordinates(cell), element_cache::AnalyticalCoefficientElementCache, time)
end

# We want this to be as fast as possible, so throw away everything unused
@inline function _assemble_element!(bₑ::AbstractVector, coords::AbstractVector{<:Vec{dim,T}}, element_cache::AnalyticalCoefficientElementCache, time) where {dim,T}
    @unpack f, cv = element_cache
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    @inbounds for (qp, w) in pairs(Ferrite.getweights(cv.qr))
        # Compute dΩ
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, qp, coords)
        dΩ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping)) * w
        # Compute x
        x = spatial_coordinate(cv, qp, coords)
        # Evaluate f
        fx = f.f(x,time)
        # TODO replace with evaluate_coefficient
        # Evaluate all basis functions
        @inbounds for j ∈ 1:getnbasefunctions(cv)
            δu = shape_value(cv, qp, j)
            bₑ[j] += fx * δu * dΩ
        end
    end
end
