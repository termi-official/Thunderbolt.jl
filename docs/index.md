# Overview

On a very high level we want the following workflow.

1. The user specifies a `Model` + `Boundary Conditions`. This also includes possible couplings.
2. The user load a compatible `Mesh`, which must hold the correct annotations for the subsets where the `Models` and `Boundary Conditions` are defined on.
3. This information defines a `Discrete Problem`.
4. The `Discrete Problem` is fed into a `Solver`, together with an `Initial Condition`.


Question: The solver is definitely responsible for managing the solution vectors, but who is responsible for setting up projectors, dof handlers and constriant handlers?
