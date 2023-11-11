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


function semidiscretize(split::ReactionDiffusionSplit{<:MonodomainModel}, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    epmodel = split.model

    ets = elementtypes(grid)
    @assert length(ets) == 1

    ip = getinterpolation(discretization.interpolations[:φₘ], getcells(grid, 1))
    dh = DofHandler(grid)
    Ferrite.add!(dh, :ϕₘ, ip)
    close!(dh);

    #
    semidiscrete_problem = SplitProblem(
        TransientHeatProblem(
            ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
            epmodel.stim,
            dh
        ),
        PointwiseODEProblem(
            # TODO epmodel.Cₘ(x) and coordinates
            ndofs(dh),
            epmodel.ion
        )
    )

    return semidiscrete_problem
end

function semidiscretize(model::StructuralModel{<:QuasiStaticModel}, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1 "Multiple elements not supported yet."

    ip = getinterpolation(discretization.interpolations[:displacement], getcells(grid, 1))
    ip_geo = Ferrite.default_geometric_interpolation(ip) # TODO get interpolation from cell
    dh = DofHandler(grid)
    Ferrite.add!(dh, :displacement, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
    close!(ch)

    semidiscrete_problem = QuasiStaticNonlinearProblem(
        dh,
        ch,
        model.mechanical_model,
        model.face_models,
    )

    return semidiscrete_problem
end

function semidiscretize(split::ReggazoniSalvadorAfricaSplit, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1 "Multiple element types not supported"
    @assert length(discretization.dbcs) == 0 "Dirichlet elimination is not supported yet."
    @assert length(split.model.base_models) == 2 "I can only handle pure mechanics coupled to pure circuit."

    semidiscrete_problem = SplitProblem(
        CoupledProblem( # Recouple mechanical problem with dummy to introduce the coupling!
            (
                semidiscretize(split.model.base_models[1], discretization, grid),
                NullProblem(1) # 1 coupling dof (chamber pressure)
            ),
            split.model.couplers
        ),
        ODEProblem(
            split.model.base_models[2],
            (du,u,t,pₗᵥ) -> lumped_driver_lv!(du, u, t, pₗᵥ[1], split.model.base_models[2]),
            [0.0] #pₗᵥ TODO better design
        )
    )

    return semidiscrete_problem
end
