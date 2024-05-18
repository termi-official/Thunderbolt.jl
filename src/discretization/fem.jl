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

    semidiscrete_problem = QuasiStaticNonlinearProblem(
        dh,
        ch,
        model.mechanical_model,
        model.face_models,
    )

    return semidiscrete_problem
end

function semidiscretize(split::RSAFDQ2022Split{<:CoupledModel}, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1 "Multiple element types not supported"
    @assert length(split.model.base_models) == 2 "I can only handle pure mechanics coupled to pure circuit."
    @error "Implementation for RSAFDQ2022Split{<:CoupledModel} currently broken. 💔"

    num_chambers = num_unknown_pressures(split.model.base_models[2])
    semidiscrete_problem = SplitProblem(
        CoupledProblem(
            (
                semidiscretize(split.model.base_models[1], discretization, grid),
                NullProblem(num_chambers) # one coupling dof for each chamber (chamber pressure)
            ),
            split.model.couplings
        ),
        ODEProblem(
            split.model.base_models[2],
            (du,u,t,chamber_pressures) -> lumped_driver!(du, u, t, chamber_pressures, split.model.base_models[2]),
            zeros(num_chambers) # Initialize with 0 pressure in the chambers
        )
    )

    return semidiscrete_problem
end

function create_chamber_tyings(coupler::LumpedFluidSolidCoupler{CVM}, structural_problem, circuit_model) where CVM
    num_unknowns_structure = solution_size(structural_problem)
    structural_problem
    chamber_tyings = RSAFDQ2022SingleChamberTying{CVM}[]
    for i in 1:length(coupler.chamber_couplings)
        # Get i-th ChamberVolumeCoupling
        coupling = coupler.chamber_couplings[i]
        # The pressure dof is just the last dof index for the structurel problem + the current chamber index
        pressure_dof_index = num_unknowns_structure + i
        chamber_faceset = getfaceset(structural_problem.dh.grid, coupling.chamber_surface_setname)
        push!(chamber_tyings, RSAFDQ2022SingleChamberTying(pressure_dof_index, chamber_faceset, coupling.chamber_volume_method, coupler.displacement_symbol))
    end
    return chamber_tyings
end

function semidiscretize(split::RSAFDQ2022Split, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1 "Multiple element types not supported"

    @unpack model = split
    @unpack structural_model, circuit_model, coupler = model
    @assert length(coupler.chamber_couplings) ≥ 1 "Provide at least one coupling for the semi-discretization of an RSAFDQ2022 model"
    @assert coupler.displacement_symbol == structural_model.displacement_symbol "Coupler is not compatible with structural model"

    # Discretize individual problems
    structural_problem = semidiscretize(model.structural_model, discretization, grid)
    num_chambers = num_unknown_pressures(model.circuit_model)

    # ODE problem for blood circuit
    flow_problem = ODEProblem(
            model.circuit_model,
        (du,u,t,chamber_pressures) -> lumped_driver!(du, u, t, chamber_pressures, model.circuit_model),
        zeros(num_chambers) # Initialize with 0 pressure in the chambers - TODO replace this hack with a proper transfer operator!
    )

    # Tie problems
    # Fix dispatch....
    chamber_tyings = create_chamber_tyings(coupler,structural_problem, circuit_model)
    @assert num_chambers == length(chamber_tyings) "Number of chambers in structural model and circuit model differs."
    semidiscrete_problem = SplitProblem(
        RSAFDQ20223DProblem(
            structural_problem,
            RSAFDQ2022TyingProblem(chamber_tyings)
        ),
        flow_problem
    )

    return semidiscrete_problem
end
