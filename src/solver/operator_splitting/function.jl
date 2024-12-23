"""
    GenericSplitFunction(functions::Tuple, solution_indices::Tuple)
    GenericSplitFunction(functions::Tuple, solution_indices::Tuple, syncronizers::Tuple)

This type of function describes a set of connected inner functions in mass-matrix form, as usually found in operator splitting procedures.

!!! note "Automatic sync"
    We should be able to get rid of the synchronizer and handle the connection of coefficients and solutions in semidiscretize.
"""
struct GenericSplitFunction{fSetType <: Tuple, idxSetType <: Tuple, sSetType <: Tuple} <: AbstractOperatorSplitFunction
    # The atomic ode functions
    functions::fSetType
    # The ranges for the values in the solution vector.
    solution_indices::idxSetType
    # Operators to update the ode function parameters
    synchronizers::sSetType
    function GenericSplitFunction(fs::Tuple, drs::Tuple, syncers::Tuple)
        @assert length(fs) == length(drs) == length(syncers)
        new{typeof(fs), typeof(drs), typeof(syncers)}(fs, drs, syncers)
    end
end

function function_size(gsf::GenericSplitFunction)
    alldofs = Set{Int}()
    for solution_indices in gsf.solution_indices
        union!(alldofs, solution_indices)
    end
    return length(alldofs)
end

num_operators(f::GenericSplitFunction) = length(f.functions)

struct NoExternalSynchronization end

GenericSplitFunction(fs::Tuple, drs::Tuple) = GenericSplitFunction(fs, drs, ntuple(_->NoExternalSynchronization(), length(fs)))

@inline get_operator(f::GenericSplitFunction, i::Integer) = f.functions[i]
@inline get_dofrange(f::GenericSplitFunction, i::Integer) = f.solution_indices[i]

recursive_null_parameters(f::AbstractOperatorSplitFunction) = @error "Not implemented"
recursive_null_parameters(f::GenericSplitFunction) = ntuple(i->recursive_null_parameters(get_operator(f, i)), length(f.functions));
recursive_null_parameters(f::DiffEqBase.AbstractDiffEqFunction) = DiffEqBase.NullParameters()
