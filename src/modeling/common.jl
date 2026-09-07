# Common modeling primitives are found here
"""
This described anything that is possibly condensed at element level.
"""
abstract type AbstractInternalModel end

struct EmptyInternalModel <: AbstractInternalModel end

# Stated for this type rather than for `AbstractInternalModel`, so that a new internal model that forgets
# to declare its state gets a `MethodError` instead of silently reporting none.
gather_internal_variable_infos(::EmptyInternalModel) = ()

struct EmptyInternalCache end

setup_internal_cache(::EmptyInternalModel, ::QuadratureRule, ::SubDofHandler) = EmptyInternalCache()

"""
    InternalVariableEvolution

Holy trait classifying the evolution law of a condensed internal variable `Q`, and with it the class
of the resulting system:

| trait                  | local problem per quadrature point | resulting system        |
| :--------------------- | :--------------------------------- | :---------------------- |
| `NoEvolution`          | none — no condensed unknown at all  | rate free               |
| `SteadyStateEvolution` | algebraic `0 = L(F, Q)`             | rate free               |
| `FirstOrderEvolution`  | `dₜQ = L(F, Q)`                     | ODE in mass matrix form |
| `RateCoupledEvolution` | `dₜQ = L(F, dₜF, Q)`                | true DAE                |

The first two rows are both rate free but are *not* interchangeable: `NoEvolution` means there is
nothing to condense, `SteadyStateEvolution` means there is a local problem to solve that happens to
carry no time derivative. They therefore select different element caches, and only the second needs a
local solver.

Note what separates `SteadyStateEvolution` from `FirstOrderEvolution`, since the names in the cache
hierarchy invite the opposite reading: `RateIndependentCondensationMaterialStateCache` means "`L`
does not read `dₜF`", not "no rate at all", and its local problem still needs a timestep.

This is a property of the *model*, deliberately not of the state cache it is lowered into. The
`Empty…CondensationMaterialStateCache` types say only that a model needs no extra scratch space for
its evaluation; they say nothing about whether it carries an internal variable or how that variable
evolves. Reading the classification off them conflates the two questions.

It is also askable before a mesh exists, which the cache-based answer is not — and that is what lets
a solver reject an incompatible model during setup rather than from the assembly loop.
"""
abstract type InternalVariableEvolution end
struct NoEvolution <: InternalVariableEvolution end
struct SteadyStateEvolution <: InternalVariableEvolution end
struct FirstOrderEvolution <: InternalVariableEvolution end
struct RateCoupledEvolution <: InternalVariableEvolution end

"""
    is_rate_free(evolution) -> Bool

Whether a local problem of this class can be posed without a timestep and a known previous state.

This is the question a continuation solver asks, and it is deliberately not `evolution isa
NoEvolution`: an algebraic constraint is condensed but rate free.
"""
is_rate_free(::InternalVariableEvolution) = false
is_rate_free(::NoEvolution) = true
is_rate_free(::SteadyStateEvolution) = true

"""
    internal_variable_evolution(model) -> InternalVariableEvolution

The [`InternalVariableEvolution`](@ref) of `model`. Material models delegate to whatever internal
model they carry, mirroring `setup_internal_cache`.
"""
internal_variable_evolution(model) = error(
    "$(typeof(model)) does not declare how its internal variable evolves. Add a method " *
    "`Thunderbolt.internal_variable_evolution(::$(typeof(model)))` returning `NoEvolution()`, " *
    "`SteadyStateEvolution()`, `FirstOrderEvolution()` or `RateCoupledEvolution()`, or delegate to " *
    "the internal model it wraps.",
)
internal_variable_evolution(::EmptyInternalModel) = NoEvolution()


abstract type AbstractSourceTerm end

"""
    is_coupling_model(model) -> Bool

Capability trait: does `model` describe a *coupling* between existing fields rather than a physics
domain of its own?

A coupling model attaches to field variables introduced by other models - typically across an
interface between subdomains - and therefore does not own a block of the solution vector the way a
bulk model does. `InterfaceDiffusionModel` is the current example.

This is deliberately independent of [`has_pointwise_reaction_part`](@ref): the two answer different
questions, and a coupling model may well carry its own reaction dynamics (e.g. gap-junction
kinetics on an interface). Code deciding *whether a model owns a domain block* must ask this trait,
not infer it from the presence or absence of a reaction part.
"""
is_coupling_model(model) = false

"""
    algebraic_variables(term) -> collection of Symbol

The scalar unknowns `term` needs in the `DofHandler` that belong to no mesh entity — a chamber
pressure acting as a Lagrange multiplier. Defaults to none.

The declaration belongs to the term that reads the unknown, so a model states its own unknowns and
`semidiscretize` needs no side channel to be told about them.
"""
algebraic_variables(term) = ()

# A model's facet terms as a tuple. A single term may be given unwrapped, and a solver-side
# annotation wrapping the collection is transparent here.
_facet_model_tuple(facet_models::Tuple) = facet_models
_facet_model_tuple(facet_model) = (facet_model,)

"""
    _model_algebraic_variables(facet_models)

The algebraic variables the facet terms declare, in `facet_models` order — which is therefore the
order they get numbered in.

A symbol declared by two terms is an error: they would share one column of the system, and nothing
says what their contributions to it should add up to.
"""
function _model_algebraic_variables(facet_models)
    names = Symbol[]
    for term in _facet_model_tuple(facet_models), name in algebraic_variables(term)
        name ∈ names && error(
            "The algebraic variable $(repr(name)) is declared by more than one facet model of the " *
            "same model. Each algebraic variable is owned by exactly one term.",
        )
        push!(names, name)
    end
    return names
end

"""
    get_time(p)

The time carried by an assembly parameter object.

Element kernels read the evaluation time from their context, `evaluation_time(args.ctx)`. This query
serves the coefficient layer, whose evaluation points are reached both from a kernel — which passes
the time it already unwrapped — and from post-processing call sites that hold a bare time and nothing
else.
"""
get_time(p) = p

include("core/heart_axes.jl")
include("core/coordinate_systems.jl")

include("core/coefficients.jl")
include("core/analytical_coefficient.jl")

include("core/weak_boundary_conditions.jl")

include("core/mass.jl")
include("core/diffusion.jl")
include("core/linear.jl")
include("core/nonlinear.jl")
include("core/multi-integrator.jl")
