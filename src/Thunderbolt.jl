module Thunderbolt

using TimerOutputs

import Unrolled: @unroll
import FastBroadcast: @..

using Reexport, UnPack
import LinearAlgebra: mul!
using SparseMatricesCSR, Polyester, LinearAlgebra
using OrderedCollections
using BlockArrays, SparseArrays, StaticArrays

using JLD2
import WriteVTK
import ReadVTK

# This is a standalone module which will be a custom package in the future
include("solver/operator_splitting.jl")
@reexport using .OS
solution_size(f::GenericSplitFunction) = OS.function_size(f)

@reexport using Ferrite
import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell, get_grid, get_coordinate_eltype
import Ferrite: vertices, edges, faces, sortedge, sortface
import Ferrite: get_coordinate_type, getspatialdim
import Ferrite: reference_shape_value

import Preferences

import Logging: Logging, LogLevel, @info, @logmsg

import SciMLBase
@reexport import SciMLBase: init, solve, solve!, step!
import DiffEqBase#: AbstractDiffEqFunction, AbstractDEProblem
import OrdinaryDiffEqCore#: OrdinaryDiffEqCore
import LinearSolve

import Base: *, +, -

import ModelingToolkit
import ModelingToolkit: @variables, @parameters, @component, @named,
        compose, ODESystem, Differential

# Accelerator support libraries
import GPUArraysCore: AbstractGPUVector, AbstractGPUArray
import Adapt:
    Adapt, adapt_structure, adapt


include("utils.jl")

include("mesh/meshes.jl")

include("ferrite-addons/InternalVariableHandler.jl")
include("ferrite-addons/transfer_operators.jl")

# Note that some modules below have an "interface.jl" but this one has only a "common.jl".
# This is simply because there is no modeling interface, but just individual physics modules and couplers.
include("modeling/common.jl")

include("modeling/microstructure.jl")

include("modeling/electrophysiology.jl")
include("modeling/solid_mechanics.jl")
include("modeling/fluid_mechanics.jl")

include("modeling/multiphysics.jl")

include("modeling/functions.jl")
include("modeling/problems.jl")

include("gpu/gpu_utils.jl")

include("discretization/interface.jl")
include("discretization/fem.jl")
include("discretization/operator.jl")

include("solver/logging.jl")
include("solver/interface.jl")
include("solver/linear.jl")
include("solver/nonlinear.jl")
include("solver/time_integration.jl")


include("modeling/electrophysiology/ecg.jl")

include("ferrite-addons/io.jl")

include("disambiguation.jl")

# TODO where to put these?
include("modeling/rsafdq2022.jl")
include("discretization/rsafdq-operator.jl")


# TODO put exports into the individual submodules above!
export
    # Coefficients
    ConstantCoefficient,
    FieldCoefficient,
    AnalyticalCoefficient,
    FieldCoefficient,
    CoordinateSystemCoefficient,
    SpectralTensorCoefficient,
    SpatiallyHomogeneousDataField,
    evaluate_coefficient,
    # Collections
    LagrangeCollection,
    DiscontinuousLagrangeCollection,
    getinterpolation,
    QuadratureRuleCollection,
    getquadraturerule,
    CellValueCollection,
    getcellvalues,
    FacetValueCollection,
    getfacevalues,
    # Mesh generators
    generate_mesh,
    generate_open_ring_mesh,
    generate_ring_mesh,
    generate_quadratic_ring_mesh,
    generate_quadratic_open_ring_mesh,
    generate_ideal_lv_mesh,
    # Generic models
    ODEProblem,
    TransientDiffusionModel,
    AffineODEFunction,
    # Local API
    PointwiseODEProblem,
    PointwiseODEFunction,
    # Mechanics
    QuasiStaticModel,
    QuasiStaticProblem,
    QuasiStaticFunction,
    PK1Model,
    PrestressedMechanicalModel,
    # Passive material models
    NullEnergyModel,
    NullCompressionPenalty,
    SimpleCompressionPenalty,
    HartmannNeffCompressionPenalty1,
    HartmannNeffCompressionPenalty2,
    HartmannNeffCompressionPenalty3,
    TransverseIsotopicNeoHookeanModel,
    HolzapfelOgden2009Model,
    LinYinPassiveModel,
    LinYinActiveModel,
    HumphreyStrumpfYinModel,
    Guccione1991PassiveModel,
    Guccione1993ActiveModel,
    LinearSpringModel,
    SimpleActiveSpring,
    # Contraction model
    ConstantStretchModel,
    PelceSunLangeveld1995Model,
    SteadyStateSarcomereModel,
    # Active model
    ActiveMaterialAdapter,
    GMKActiveDeformationGradientModel,
    GMKIncompressibleActiveDeformationGradientModel,
    RLRSQActiveDeformationGradientModel,
    SimpleActiveStress,
    PiersantiActiveStress,
    # Electrophysiology
    MonodomainModel,
    # ParabolicParabolicBidomainModel,
    # ParabolicEllipticBidomainModel,
    NoStimulationProtocol,
    TransmembraneStimulationProtocol,
    AnalyticalTransmembraneStimulationProtocol,
    ReactionDiffusionSplit,
    # Circuit
    RSAFDQ2022LumpedCicuitModel,
    MTKLumpedCicuitModel,
    # FSI
    RSAFDQ2022Model,
    RSAFDQ2022SurrogateVolume,
    RSAFDQ2022Split,
    Hirschvogel2017SurrogateVolume,
    LumpedFluidSolidCoupler,
    ChamberVolumeCoupling,
    VolumeTransfer0D3D,
    PressureTransfer3D0D,
    # Microstructure
    AnisotropicPlanarMicrostructureModel,
    AnisotropicPlanarMicrostructure,
    OrthotropicMicrostructureModel,
    OrthotropicMicrostructure,
    TransverselyIsotropicMicrostructureModel,
    TransverselyIsotropicMicrostructure,
    create_simple_microstructure_model,
    # Coordinate system
    LVCoordinateSystem,
    LVCoordinate,
    BiVCoordinate,
    CartesianCoordinateSystem,
    compute_lv_coordinate_system,
    compute_midmyocardial_section_coordinate_system,
    getcoordinateinterpolation,
    vtk_coordinate_system,
    # Coupling
    Coupling,
    CoupledModel,
    # Discretization
    semidiscretize,
    FiniteElementDiscretization,
    # Solver
    SchurComplementLinearSolver,
    NewtonRaphsonSolver,
    HomotopyPathSolver,
    ForwardEulerSolver,
    BackwardEulerSolver,
    ForwardEulerCellSolver,
    AdaptiveForwardEulerSubstepper,
    # Integrator
    # Utils
    calculate_volume_deformed_mesh,
    elementtypes,
    QuadraturePoint,
    QuadratureIterator,
    load_carp_grid,
    load_voom2_grid,
    load_mfem_grid,
    solution_size,
    # IO
    ParaViewWriter,
    JLD2Writer,
    store_timestep!,
    store_timestep_celldata!,
    store_timestep_field!,
    store_coefficient!,
    store_green_lagrange!,
    finalize_timestep!,
    finalize!,
    # Mechanical PDEs
    GeneralizedHillModel,
    ActiveStressModel,
    ExtendedHillModel,
    #  BCs
    NormalSpringBC,
    PressureFieldBC,
    BendingSpringBC,
    RobinBC,
    ConstantPressureBC
end
