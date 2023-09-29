"""
"""
struct FiniteElementDiscretization
    """
    """
    interpolations::Dict{Symbol, Thunderbolt.InterpolationCollection}
    """
    """
    dbcs::Vector{Dirichlet}
    """
    """
    function FiniteElementDiscretization(ips::Dict{Symbol, Thunderbolt.InterpolationCollection})
        new(ips, Dirichlet[])
    end
end

