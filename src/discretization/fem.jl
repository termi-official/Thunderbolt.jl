"""
"""
struct FiniteElementDiscretization
    """
    """
    interpolations::Dict{Symbol, InterpolationCollection}
    """
    """
    dbcs::Vector{Dirichlet}
    """
    """
    function FiniteElementDiscretization(ips::Dict{Symbol, <: InterpolationCollection})
        new(ips, Dirichlet[])
    end
    
    function FiniteElementDiscretization(ips::Dict{Symbol, <: InterpolationCollection}, dbcs::Vector{Dirichlet})
        new(ips, dbcs)
    end
end

