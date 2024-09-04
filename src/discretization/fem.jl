"""
Descriptor for a finite element discretization of a part of a PDE over some subdomain.
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
    subdomains::Vector{String}
    """
    """
    function FiniteElementDiscretization(ips::Dict{Symbol, <: InterpolationCollection}, dbcs::Vector{Dirichlet} = Dirichlet[], subdomains::Vector{String} = [""])
        new(ips, dbcs, subdomains)
    end
end

# Internal utility with proper error message
function _get_interpolation_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    if !haskey(disc.interpolations, sym)
        error("Finite element discretization does not have an interpolation for $sym. Available symbols: $(collect(keys(disc.interpolations))).")
    end
    return disc.interpolations[sym]
end

semidiscretize(::CoupledModel, discretization, mesh::AbstractGrid) = @error "No implementation for the generic discretization of coupled problems available yet."

function semidiscretize(model::TransientDiffusionModel, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    @assert length(discretization.dbcs) == 0 "Dirichlet conditions not supported yet for TransientDiffusionProblem"

    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

    return TransientDiffusionFunction(
        model.κ,
        model.source,
        dh
    )
end

function semidiscretize(model::SteadyDiffusionModel, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
    close!(ch)

    return SteadyDiffusionFunction(
        model.κ,
        model.source,
        dh,
        ch
    )
end

function semidiscretize(split::ReactionDiffusionSplit{<:MonodomainModel}, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    epmodel = split.model
    φsym = epmodel.transmembrane_solution_symbol

    heat_model = TransientDiffusionModel(
        ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
        epmodel.stim,
        φsym,
    )

    heatfun = semidiscretize(
        heat_model,
        discretization,
        mesh,
    )

    dh = heatfun.dh
    ndofsφ = ndofs(dh)
    # TODO we need some information about the discretization of this one, e.g. dofs a nodes vs dofs at quadrature points
    # TODO we should call semidiscretize here too - This is a placeholder for the nodal discretization
    odefun = PointwiseODEFunction(
        # TODO epmodel.Cₘ(x)
        ndofsφ,
        epmodel.ion,
        split.cs === nothing ? nothing : compute_nodal_values(split.cs, dh, φsym)
    )
    nstates_per_point = num_states(odefun.ode)
    # TODO this assumes that the transmembrane potential is the first field. Relax this.
    heat_dofrange = 1:ndofsφ
    ode_dofrange = 1:nstates_per_point*ndofsφ
    #
    semidiscrete_ode = GenericSplitFunction(
        (heatfun, odefun),
        (heat_dofrange, ode_dofrange),
        # No transfer operators needed, because the the solutions variables overlap with the subproblems perfectly
    )

    return semidiscrete_ode
end

function semidiscretize(model::StructuralModel{<:QuasiStaticModel}, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    sym = model.displacement_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

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
