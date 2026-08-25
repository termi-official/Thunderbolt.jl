"""
    _chamber_coupling(dh, displacement_symbol, chamber)

The sparsity the chamber pressure needs, as Ferrite's coupling descriptor.

`CellCoupling` over the whole `DofHandler` is what the shared `global_dofs` declaration requires:
the pressure sits in the tail of *every* element-local system of the subdomains carrying the tying
term, so the cell sweep scatters through the coupling entries of every cell -- even where the
element writes zeros into them. Narrowing this to the endocardial surface needs a per-item-family
`global_dofs` declaration, which FerriteOperators does not offer.
"""
_chamber_coupling(dh, displacement_symbol, chamber) = CellCoupling(
    collect(Int, Iterators.flatten(sdh.cellset for sdh in dh.subdofhandlers));
    algebraic_coupling = ((displacement_symbol, chamber.pressure_symbol),),
)

function setup_stage_operator(
    f::RSAFDQ20223DFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
)
    (; tying_info, structural_function) = f
    (; dh, ch, integrator) = structural_function
    chambers = tying_info.chambers
    n_chambers = length(chambers)
    n_u = ndofs(dh) - n_chambers

    # The tying facets write into one dof shared by every chamber facet, which no coloring can make
    # race free, so the scheduling is sequential regardless of what the discretization asked for.
    couplings = Tuple(
        _chamber_coupling(dh, chamber.displacement_symbol, chamber) for chamber in chambers
    )
    # CSC blocks, not the CSR of FerriteOperators' own blocked-assembly example:
    # `SchurComplementLinearSolver`'s inner `UMFPACKFactorization` factorizes the (1,1) block, which
    # needs CSC.
    strategy = AssemblyStrategy(
        FullAssembly(
            FerriteOperators.BlockedOperatorSpecification(
                [n_u, n_chambers],
                BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}};
                algebraic_couplings = couplings,
                constraint_handler = ch,
            ),
        ),
        SequentialScheduling(),
        get_strategy(f).device,
    )

    return setup_operator(strategy, integrator, dh; slots = THUNDERBOLT_STAGE_SLOTS)
end
