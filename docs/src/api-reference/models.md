```@meta
DocTestSetup = :(using Thunderbolt)
```

# Models

## Coefficient

```@docs
ConstantCoefficient
FieldCoefficient
AnalyticalCoefficient
SpectralTensorCoefficient
SpatiallyHomogeneousDataField
CoordinateSystemCoefficient
evaluate_coefficient
```

## Microstructure

```@docs
AnisotropicPlanarMicrostructureModel
OrthotropicMicrostructureModel
create_simple_microstructure_model
Thunderbolt.streeter_type_fsn
```

## Boundary Conditions

```@docs
RobinBC
NormalSpringBC
BendingSpringBC
ConstantPressureBC
PressureFieldBC
```

## Solid Mechanics

```@docs
StructuralModel
ExtendedHillModel
GeneralizedHillModel
ActiveStressModel
```

### Passive Energies

```@docs
NullEnergyModel
LinearSpringModel
TransverseIsotopicNeoHookeanModel
HolzapfelOgden2009Model
LinYinPassiveModel
LinYinActiveModel
HumphreyStrumpfYinModel
Guccione1991PassiveModel
Thunderbolt.BioNeoHooekean
```

### Active Energies

```@docs
SimpleActiveSpring
ActiveMaterialAdapter
```

### Active Deformation Gradients

```@docs
GMKActiveDeformationGradientModel
GMKIncompressibleActiveDeformationGradientModel
RLRSQActiveDeformationGradientModel
```

### Active Stresses

```@docs
SimpleActiveStress
PiersantiActiveStress
Guccione1993ActiveModel
```

### Compression

```@docs
NullCompressionPenalty
SimpleCompressionPenalty
HartmannNeffCompressionPenalty1
HartmannNeffCompressionPenalty2
HartmannNeffCompressionPenalty3
```

## Electrophysiology

```@docs
MonodomainModel
ParabolicParabolicBidomainModel
ParabolicEllipticBidomainModel
ReactionDiffusionSplit
```

```@docs
NoStimulationProtocol
TransmembraneStimulationProtocol
AnalyticalTransmembraneStimulationProtocol
```

## Cells

!!! warning
    These are intended to be replaced by ModelingToolkit analogues!

```@docs
Thunderbolt.ParametrizedFHNModel
Thunderbolt.ParametrizedPCG2019Model
```


## Fluid Mechanics

### Lumped Models

```@docs
ReggazoniSalvadorAfricaLumpedCicuitModel
```

## Multiphysics

### Interface

```@docs
Thunderbolt.InterfaceCoupler
Thunderbolt.VolumeCoupler
Coupling
CoupledModel
```

### FSI

```@docs
LumpedFluidSolidCoupler
Hirschvogel2017SurrogateVolume
ReggazoniSalvadorAfrica2022SurrogateVolume
ReggazoniSalvadorAfricaSplit
```
