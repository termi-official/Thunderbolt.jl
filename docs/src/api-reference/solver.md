```@meta
DocTestSetup = :(using Thunderbolt)
```

# Solver

## Linear

```@docs
SchurComplementLinearSolver
```

## Preconditioners

```@docs
Thunderbolt.Preconditioners.L1GSPrecBuilder
Thunderbolt.Preconditioners.L1GSPreconditioner
Thunderbolt.Preconditioners.ForwardSweep
Thunderbolt.Preconditioners.BackwardSweep
Thunderbolt.Preconditioners.SymmetricSweep
Thunderbolt.Preconditioners.PackedBufferCache
Thunderbolt.Preconditioners.MatrixViewCache
```

## Nonlinear

```@docs
NewtonRaphsonSolver
MultiLevelNewtonRaphsonSolver
Thunderbolt.AbstractStageFunction
Thunderbolt.update_stage_linearization!
Thunderbolt.evaluate_stage_residual!
Thunderbolt.condense_stage!
```


## Time

```@docs
BackwardEulerSolver
ForwardEulerCellSolver
AdaptiveForwardEulerSubstepper
HomotopyPathSolver
NewmarkSolver
```

## Operator Splitting Adaptivity

```@docs
Thunderbolt.ReactionTangentController
```

## Step size control

```@docs
Thunderbolt.PIDController
Thunderbolt.adaptive_order
Thunderbolt.set_error_estimate!
Thunderbolt.velocity
Thunderbolt.acceleration
```
