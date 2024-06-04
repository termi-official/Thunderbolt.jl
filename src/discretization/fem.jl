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

semidiscretize(::CoupledModel, discretization, grid) = @error "No implementation for the generic discretization of coupled problems available yet."

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
        TransientHeatFunction(
            ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
            epmodel.stim,
            dh
        ),
        PointwiseODEFunction(
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

    ip = getinterpolation(discretization.interpolations[model.displacement_symbol], getcells(grid, 1))
    ip_geo = Ferrite.default_geometric_interpolation(ip) # TODO get interpolation from cell
    dh = DofHandler(grid)
    Ferrite.add!(dh, model.displacement_symbol, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
    close!(ch)

    semidiscrete_problem = QuasiStaticNonlinearFunction(
        dh,
        ch,
        model.mechanical_model,
        model.face_models
    )

    return semidiscrete_problem
end

function create_chamber_tyings(coupler::LumpedFluidSolidCoupler{CVM}, structural_problem, circuit_model) where CVM
    num_unknowns_structure = solution_size(structural_problem)
    chamber_tyings = RSAFDQ2022SingleChamberTying{CVM}[]
    for i in 1:length(coupler.chamber_couplings)
        # Get i-th ChamberVolumeCoupling
        coupling = coupler.chamber_couplings[i]
        # The pressure dof is just the last dof index for the structurel problem + the current chamber index
        pressure_dof_index = num_unknowns_structure + i
        chamber_faceset = getfaceset(structural_problem.dh.grid, coupling.chamber_surface_setname)
        chamber_volume_idx_lumped = get_variable_symbol_index(circuit_model, coupling.lumped_model_symbol)
        initial_volume_lumped = NaN # We do this to catch initializiation issues downstream
        tying = RSAFDQ2022SingleChamberTying(
            pressure_dof_index,
            pressure_dof_index,
            chamber_faceset,
            coupling.chamber_volume_method,
            coupler.displacement_symbol,
            initial_volume_lumped,
            chamber_volume_idx_lumped,
            num_unknowns_structure+num_unknown_pressures(circuit_model)+chamber_volume_idx_lumped,
        )
        push!(chamber_tyings, tying)
    end
    return chamber_tyings
end
