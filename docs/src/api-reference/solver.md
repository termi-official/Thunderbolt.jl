```@meta
DocTestSetup = :(using Thunderbolt)
```

# Solver

## Linear

```@docs
SchurComplementLinearSolver
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
AdaptiveForwardEulerSubstepper
LoadDrivenSolver
```

## Operator Splitting Module

```@docs
Thunderbolt.OS.LieTrotterGodunov
Thunderbolt.OS.GenericSplitFunction
Thunderbolt.OS.OperatorSplittingIntegrator
```

## Operator Splitting Adaptivity

```@docs
Thunderbolt.AdaptiveOperatorSplittingAlgorithm
Thunderbolt.ReactionTangentController
Thunderbolt.get_reaction_tangent
Thunderbolt.OS.stepsize_controller!
Thunderbolt.OS.update_dt!
Thunderbolt.get_next_dt
```
