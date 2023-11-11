# Overview

On a very high level we want the following workflow.

1. The user specifies a `Model` + `Boundary Conditions`. This also includes possible couplings.
2. The user load a compatible `Mesh`, which must hold the correct annotations for the subsets where the `Models` and `Boundary Conditions` are defined on.
3. This information defines a `Discrete Problem`.
4. The `Discrete Problem` is fed into a `Solver`, together with an `Initial Condition`.


Question: The solver is definitely responsible for managing the solution vectors, but who is responsible for setting up projectors, dof handlers and constriant handlers?


# Design details

## Modeling

Questions:
1. How to pass and solve state information for internal variables around?

## Operators

`assemble_element`

Questions:
1. How to deal with quasi-static problems which also need velocities?
2. How to make recursive assemble_element definitions for volume coupled problems?

## Solver

`setup_solver_caches(problem, solver, t₀)` takes the problem and a solver to setup the operators needed during solve.

Questions: 
1. How to control which exact operator?
1. When and how to check if the operator is compatible with the solver?

## Benchmarking

To investiage the performance we can use the following code snippet, which should be self-explanatory

```julia
using Thunderbolt.TimerOutputs
TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
run_simulation()
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)
```

It makes sense to make sure the code is properly precompiled before benchmarkins, e.g. by calling `run_simulation()` once before running the code snippet.
