# Developer documentation

More devdocs coming soon.


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

See also [my brain flushing in real time some design decisions](DifferentialEquationsjl-issue.md).
