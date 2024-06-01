module Thunderbolt

using TimerOutputs

using Reexport, UnPack
import LinearAlgebra: mul!
using SparseMatricesCSR, Polyester, LinearAlgebra
using Krylov
using OrderedCollections
@reexport using Ferrite
using BlockArrays, SparseArrays, StaticArrays

using JLD2

import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell
import Ferrite: vertices, edges, faces, sortedge, sortface

import DiffEqBase#: AbstractDiffEqFunction, AbstractDEProblem

import Krylov: CgSolver

import Base: *, +, -

import CommonSolve: init, solve, solve!, step!

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

include("modeling/problems.jl") # This is not really "modeling" but a glue layer to translate from model to solver via a discretization

include("discretization/interface.jl")
include("discretization/fem.jl")
include("discretization/operator.jl")

include("solver/interface.jl")
include("solver/integrator.jl")
include("solver/newton_raphson.jl")
include("solver/load_stepping.jl")
include("solver/euler.jl")
include("solver/partitioned_solver.jl")
include("solver/operator_splitting.jl")

# FIXME move into integrator
function DiffEqBase.step!(integrator::ThunderboltIntegrator, dt, stop_at_tdt = false)
    # OridinaryDiffEq lets dt be negative if tdir is -1, but that's inconsistent
    dt <= zero(dt) && error("dt must be positive")
    # stop_at_tdt && !integrator.dtchangeable && error("Cannot stop at t + dt if dtchangeable is false")
    t_plus_dt = integrator.t + dt
    stop_at_tdt && DiffEqBase.add_tstop!(integrator, t_plus_dt)
    while !OS.reached_tstop(integrator, t_plus_dt, stop_at_tdt)
        # Solve inner problem
        perform_step!(integrator.f, integrator.cache, integrator.t, integrator.dt)
        integrator.tprev = integrator.t
        integrator.t = integrator.t + integrator.dt
    end
end
function OS.synchronize_subintegrator!(subintegrator::ThunderboltIntegrator, integrator::OS.OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    subintegrator.t = t
    subintegrator.dt = dt
end
OS.tdir(::ThunderboltIntegrator) = 1 # TODO remove

include("solver/ecg.jl")

include("io.jl")

include("disambiguation.jl")

# TODO put exports into the individual submodules above!
export
    legacysolve,
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
    FaceValueCollection,
    getfacevalues,
    # Mesh generators
    generate_mesh,
    generate_open_ring_mesh,
    generate_ring_mesh,
    generate_quadratic_ring_mesh,
    generate_quadratic_open_ring_mesh,
    generate_ideal_lv_mesh,
    # Mechanics
    StructuralModel,
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
    ParabolicParabolicBidomainModel,
    ParabolicEllipticBidomainModel,
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
    solve!,
    NewtonRaphsonSolver,
    LoadDrivenSolver,
    ForwardEulerSolver,
    BackwardEulerSolver,
    ForwardEulerCellSolver,
    LTGOSSolver,
    # Utils
    default_initializer,
    calculate_volume_deformed_mesh,
    elementtypes,
    QuadraturePoint,
    QuadratureIterator,
    load_carp_mesh,
    load_voom2_mesh,
    load_mfem_mesh,
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
