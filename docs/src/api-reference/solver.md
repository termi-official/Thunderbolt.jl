```@meta
DocTestSetup = :(using Thunderbolt)
```

# Solver

## Linear

```@docs
Thunderbolt.Schur2x2SaddleFormLinearSolver
```

## Nonlinear

```@docs
NewtonRaphsonSolver
```


## Time

```@docs
BackwardEulerSolver
ForwardEulerSolver
ForwardEulerCellSolver
Thunderbolt.AdaptiveForwardEulerReactionSubCellSolver
Thunderbolt.ThreadedForwardEulerCellSolver
LoadDrivenSolver
```

## Operator Splitting Module

```@docs
Thunderbolt.OS.LieTrotterGodunov
Thunderbolt.OS.GenericSplitFunction
Thunderbolt.OS.OperatorSplittingIntegrator
```
