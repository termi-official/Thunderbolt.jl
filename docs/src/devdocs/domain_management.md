# Domain management

Having multiple coupled subdomains is very common in multiphyics problems.
Furthermore it is also not uncommon to have mixed(-dimensional) grids, think e.g. about the Purkinje network and the myocardium in chamber electrophysiology simulations.
To manage these cases Thunderbolt.jl comes with some utilities. The first one is the [`SimpleMesh`](@ref), which is takes a [`Ferrite.Grid`] and extracts information about the subdomains. The subdomains are split up by element type to handle mixed grids properly.

This subdomain information can then be used to construct [`Ferrite.SubDofHandler`] to manage the field variables on subdomains:

```@docs
Thunderbolt.add_subdomain!
Thunderbolt.ApproximationDescriptor
```

Furthermore to manage data on subdomains we provide a non-uniform matrix-like data type.

```@docs
Thunderbolt.DenseDataRange
Thunderbolt.get_data_for_index
```

Two examples where this is used: The storate of element assembly and quadrature data on mixed grids.

## Multidomain Assembly

The operators in Thunderbolt work very similar w.r.t. the management of multiple domains. The all follow the following pattern:

```julia
function update_operator(op, time)
    # Sanity check to see if the operator is internally consistent
    # e.g. are all fields are present in the associated dof handlers, ...
    check_internal_correctness(op)
    # Depending on the operator and matrix type we get the correct assembler
    assembler = instantiate_specific_assembler(op)
    for sdh in op.dh.subdofhandlers
        # We create a new or get a from the operator some scratch to make the assembly loop allocation free
        # and possibly to precompute some generic stuff
        weak_form_cache = setup_or_query_cache_for_subdomain(op, sdh)
        # This step also acts a function barrier and contains the hot loop over the elements on the subdomain
        # which actually fills the matrix
        assemble_on_subdomain!(assembler, sdh, weak_form_cache)
    end
    # Some assemblers need a finalization step, e.g. distributed assembly, COO assembly, EA collapse, ...
    finalize_assembly(assembler)
```
