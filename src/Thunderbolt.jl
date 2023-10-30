module Thunderbolt

using Reexport, UnPack, StaticArrays
import LinearAlgebra: mul!
using SparseMatricesCSR
using Krylov
using OrderedCollections
@reexport using Ferrite
using BlockArrays

using JLD2

import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell
import Ferrite: vertices, edges, faces, sortedge, sortface

import Krylov: CgSolver

include("collections.jl")
include("utils.jl")

include("mesh/meshes.jl")
include("mesh/coordinate_systems.jl")
include("mesh/tools.jl")
include("mesh/generators.jl")

include("modeling/coefficients.jl")

include("modeling/boundary_conditions.jl")

include("modeling/microstructure.jl")

include("modeling/electrophysiology.jl")

include("modeling/solid/energies.jl")
include("modeling/solid/contraction.jl")
include("modeling/solid/active.jl")
include("modeling/solid/drivers.jl") # TODO better name. This is basically the quadrature point routine.

include("modeling/fluid/lumped.jl")

include("modeling/coupler/interface.jl")
include("modeling/coupler/fsi.jl")

include("modeling/problems.jl")

include("solver/interface.jl")
include("solver/operator.jl")
include("solver/newton_raphson.jl")
include("solver/load_stepping.jl")
include("solver/backward_euler.jl")
include("solver/partitioned_solver.jl")
include("solver/operator_splitting.jl")

include("discretization/interface.jl")
include("discretization/fem.jl")


include("io.jl")

export
    solve,
    # Coefficients
    ConstantCoefficient,
    FieldCoefficient,
    AnalyticalCoefficient,
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
    QuasiStaticModel,
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
    # Microstructure
    OrthotropicMicrostructureModel,
    directions,
    FieldCoefficient,
    ConstantCoefficient,
    evaluate_coefficient,
    create_simple_fiber_model,
    update_microstructure_cache!,
    setup_microstructure_cache,
    # Coordinate system
    LVCoordinateSystem,
    CartesianCoordinateSystem,
    compute_LV_coordinate_system,
    compute_midmyocardial_section_coordinate_system,
    getcoordinateinterpolation,
    vtk_coordinate_system,
    # Discretization
    semidiscretize,
    FiniteElementDiscretization,
    # Solver
    setup_solver_caches,
    solve!,
    NewtonRaphsonSolver,
    LoadDrivenSolver,
    BackwardEulerSolver,
    ForwardEulerCellSolver,
    LTGOSSolver,
    ReactionDiffusionSplit,
    # Utils
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
    NormalSpringBC
end
