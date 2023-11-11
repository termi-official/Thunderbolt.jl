module Thunderbolt

using Reexport, UnPack
import LinearAlgebra: mul!
using SparseMatricesCSR
using Krylov
using OrderedCollections
@reexport using Ferrite
using BlockArrays, SparseArrays, StaticArrays

using JLD2

import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell
import Ferrite: vertices, edges, faces, sortedge, sortface

import Krylov: CgSolver

include("utils.jl")

include("mesh/meshes.jl")

# Note that some modules below have an "interface.jl" but this one has only a "common.jl".
# This is simply because there is no modeling interface, but just individual physics modules and couplers.
include("modeling/common.jl")

include("modeling/microstructure.jl")

include("modeling/electrophysiology.jl")
include("modeling/solid_mechanics.jl")
include("modeling/fluid_mechanics.jl")

include("modeling/multiphysics.jl")

include("modeling/problems.jl") # This is not really "modeling" but a glue layer to translate from model to solver via a discretization

include("solver/interface.jl")
include("solver/operator.jl")
include("solver/newton_raphson.jl")
include("solver/load_stepping.jl")
include("solver/euler.jl")
include("solver/partitioned_solver.jl")
include("solver/operator_splitting.jl")

include("discretization/interface.jl")
include("discretization/fem.jl")

include("io.jl")

# TODO put exports into the individual submodules above!
export
    solve,
    # Coefficients
    ConstantCoefficient,
    FieldCoefficient,
    AnalyticalCoefficient,
    FieldCoefficient,
    evaluate_coefficient,
    # Collections
    LagrangeCollection,
    getinterpolation,
    QuadratureRuleCollection,
    getquadraturerule,
    CellValueCollection,
    FaceValueCollection,
    # Mesh generators
    generate_mesh,
    generate_ring_mesh,
    # Mechanics
    StructuralModel,
    # Passive material models
    NullEnergyModel,
    NullCompressionPenalty,
    SimpleCompressionPenalty,
    NeffCompressionPenalty,
    TransverseIsotopicNeoHookeanModel,
    HolzapfelOgden2009Model,
    LinYinPassiveModel,
    LinYinActiveModel,
    HumphreyStrumpfYinModel,
    Guccione1991PassiveModel,
    Guccione1993ActiveModel,
    LinearSpringModel,
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
    ParabolicParabolicBidomainModel,
    ParabolicEllipticBidomainModel,
    NoStimulationProtocol,
    # Circuit
    ReggazoniSalvadorAfricaLumpedCicuitModel,
    # FSI
    Hirschvogel2016SurrogateVolume,
    LumpedFluidSolidCoupler,
    # Microstructure
    AnisotropicPlanarMicrostructureModel,
    OrthotropicMicrostructureModel,
    create_simple_fiber_model,
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
    solve!,
    NewtonRaphsonSolver,
    LoadDrivenSolver,
    ForwardEulerSolver,
    BackwardEulerSolver,
    ForwardEulerCellSolver,
    LTGOSSolver,
    ReactionDiffusionSplit,
    ReggazoniSalvadorAfricaSplit,
    # Utils
    default_initializer,
    calculate_volume_deformed_mesh,
    elementtypes,
    QuadraturePoint,
    # IO
    ParaViewWriter,
    JLD2Writer,
    store_timestep!,
    finalize!,
    # Mechanical PDEs
    GeneralizedHillModel,
    ActiveStressModel,
    ExtendedHillModel,
    #  BCs
    NormalSpringBC,
    PressureFieldBC,
    ConstantPressureBC
end
