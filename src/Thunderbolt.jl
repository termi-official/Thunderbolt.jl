module Thunderbolt

using Reexport, UnPack, StaticArrays
import LinearAlgebra: mul!
using SparseMatricesCSR
using Krylov
using OrderedCollections
@reexport using Ferrite

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

include("modeling/mechanics/energies.jl")
include("modeling/mechanics/contraction.jl")
include("modeling/mechanics/active.jl")
include("modeling/mechanics/drivers.jl") # TODO better name. This is basically the quadrature point routine.

include("modeling/problems.jl")

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
    CalciumHatField,
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
    LazyCoefficientCache,
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
    value,
    # IO
    ParaViewWriter,
    JLD2Writer,
    store_timestep!,
    finalize!,
    # Drivers
    GeneralizedHillModel,
    ActiveStressModel,
    ExtendedHillModel,
    constitutive_driver,
    #  BCs
    NormalSpringBC

end
