```@meta
DocTestSetup = :(using Thunderbolt)
```

# Solver

```@docs
BackwardEulerSolver
ForwardEulerSolver
ForwardEulerCellSolver
Thunderbolt.AdaptiveForwardEulerReactionSubCellSolver
Thunderbolt.ThreadedForwardEulerCellSolver
LoadDrivenSolver
NewtonRaphsonSolver
```

## Operator Splitting Module

```@docs
Thunderbolt.OS.LieTrotterGodunov
Thunderbolt.OS.GenericSplitFunction
Thunderbolt.OS.OperatorSplittingIntegrator
```
