module Thunderbolt

using TimerOutputs

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
# include("solver/local_time_stepping.jl")
# include("solver/multilevel.jl")

@reexport using Ferrite
import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell, get_grid
import Ferrite: vertices, edges, faces, sortedge, sortface

import DiffEqBase#: AbstractDiffEqFunction, AbstractDEProblem
@reexport import LinearSolve

import Base: *, +, -

@reexport import CommonSolve: init, solve, solve!, step!

import ModelingToolkit
import ModelingToolkit: @variables, @parameters, @component, @named,
        compose, ODESystem, Differential

include("utils.jl")

include("mesh/meshes.jl")

include("transfer_operators.jl")

# Note that some modules below have an "interface.jl" but this one has only a "common.jl".
# This is simply because there is no modeling interface, but just individual physics modules and couplers.
include("modeling/common.jl")

include("modeling/microstructure.jl")

include("modeling/electrophysiology.jl")
include("modeling/solid_mechanics.jl")
include("modeling/fluid_mechanics.jl")

include("modeling/multiphysics.jl")

include("modeling/functions.jl")
include("modeling/problems.jl") # Utility for compat against DiffEqBase

include("discretization/interface.jl")
include("discretization/fem.jl")
include("discretization/operator.jl")

include("solver/interface.jl")
include("solver/linear.jl")
include("solver/nonlinear.jl")
include("solver/time_integration.jl")

include("processing/ecg.jl")

include("io.jl")

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
    TransientHeatModel,
    TransientHeatFunction,
    # Local API
    PointwiseODEProblem,
    PointwiseODEFunction,
    # Mechanics
    StructuralModel,
    QuasiStaticProblem,
    QuasiStaticNonlinearFunction,
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
    OrthotropicMicrostructureModel,
    create_simple_microstructure_model,
    # Coordinate system
    LVCoordinateSystem,
    CartesianCoordinateSystem,
    compute_LV_coordinate_system,
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
    LoadDrivenSolver,
    ForwardEulerSolver,
    BackwardEulerSolver,
    ForwardEulerCellSolver,
    AdaptiveForwardEulerSubstepper,
    # Integrator
    get_parent_index,
    get_parent_value,
    # Utils
    calculate_volume_deformed_mesh,
    elementtypes,
    QuadraturePoint,
    QuadratureIterator,
    load_carp_mesh,
    load_voom2_mesh,
    load_mfem_mesh,
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
