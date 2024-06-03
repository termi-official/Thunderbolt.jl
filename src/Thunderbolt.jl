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
    dt <= zero(dt) && error("dt must be positive")
    tnext = integrator.t + dt
    while !OS.reached_tstop(integrator, tnext, stop_at_tdt)
        # Solve inner problem
        perform_step!(integrator, integrator.cache)
        # TODO check for solver failure
        # Update integrator
        integrator.tprev = integrator.t
        integrator.t = integrator.t + integrator.dt
    end
end
function OS.synchronize_subintegrator!(subintegrator::ThunderboltIntegrator, integrator::OS.OperatorSplittingIntegrator)
    @unpack t, dt = integrator
    subintegrator.t = t
    subintegrator.dt = dt
end
# TODO some operator splitting methods require to go back in time, so we need to figure out what the best way is.
OS.tdir(::ThunderboltIntegrator) = 1

# TODO Any -> cache supertype
function OS.advance_solution_to!(integrator::ThunderboltIntegrator, cache::Any, tend)
    @unpack f, t = integrator
    dt = tend-t
    DiffEqBase.step!(integrator, dt, true)
end
@inline function OS.prepare_local_step!(subintegrator::ThunderboltIntegrator)
    # Copy solution into subproblem
    # uparentview = @view subintegrator.uparent[subintegrator.indexset]
    # subintegrator.u .= subintegrator.uparent[subintegrator.indexset]
    for (i,imain) in enumerate(subintegrator.indexset)
        subintegrator.u[i] = subintegrator.uparent[imain]
    end
    # Mark previous solution
    subintegrator.uprev .= subintegrator.u
    syncronize_parameters!(subintegrator, subintegrator.f, subintegrator.synchronizer)
end
@inline function OS.finalize_local_step!(subintegrator::ThunderboltIntegrator)
    # Copy solution out of subproblem
    #
    # uparentview = @view subintegrator.uparent[subintegrator.indexset]
    # uparentview .= subintegrator.u
    for (i,imain) in enumerate(subintegrator.indexset)
        subintegrator.uparent[imain] = subintegrator.u[i]
    end
    # TODO
end
# Glue code
function OS.build_subintegrators_recursive(f, p::Any, cache::AbstractTimeSolverCache, u::AbstractArray, uprev::AbstractArray, t, dt)
    return Thunderbolt.ThunderboltIntegrator(f, cache.uₙ, cache.uₙ₋₁, p, t, dt)
end
function OS.build_subintegrators_recursive(f, synchronizer, p::Any, cache::AbstractTimeSolverCache, u::AbstractArray, uprev::AbstractArray, t, dt, dof_range, uparent)
    integrator = Thunderbolt.ThunderboltIntegrator(f, cache.uₙ, uparent, cache.uₙ₋₁, dof_range, p, t, t, dt, cache, synchronizer, nothing, true)
    # This makes sure that the parameters are set correctly for the first time step
    syncronize_parameters!(integrator, f, synchronizer)
    return integrator
end
function OS.construct_inner_cache(f, alg::AbstractSolver, u::AbstractArray, uprev::AbstractArray)
    return Thunderbolt.setup_solver_cache(f, alg, 0.0)
end
OS.recursive_null_parameters(stuff::AbstractSemidiscreteProblem) = OS.DiffEqBase.NullParameters()
syncronize_parameters!(integ, f, ::OS.NoExternalSynchronization) = nothing



"""
    Utility function to synchronize the volume in a split [`RSAFDQ2022Problem`](@ref)
"""
struct VolumeTransfer0D3D{TP <: Thunderbolt.RSAFDQ2022TyingProblem} <: Thunderbolt.AbstractTransferOperator
    tying::TP
end

function Thunderbolt.syncronize_parameters!(integ, f, syncer::VolumeTransfer0D3D)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for chamber ∈ syncer.tying.chambers
        chamber.V⁰ᴰval = integ.uparent[chamber.V⁰ᴰidx_global]
    end
end

"""
    Utility function to synchronize the pressire in a split [`RSAFDQ2022Problem`](@ref)
"""
struct PressureTransfer3D0D{TP <: Thunderbolt.RSAFDQ2022TyingProblem} <: Thunderbolt.AbstractTransferOperator
    tying::TP
end

function Thunderbolt.syncronize_parameters!(integ, f, syncer::PressureTransfer3D0D)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for (chamber_idx,chamber) ∈ enumerate(syncer.tying.chambers)
        p = integ.uparent[chamber.pressure_dof_index_global]
        # The pressure buffer is constructed in a way that the chamber index and
        # pressure index coincides
        f.p[chamber_idx] = p
    end
end



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
    solve!,
    NewtonRaphsonSolver,
    LoadDrivenSolver,
    ForwardEulerSolver,
    BackwardEulerSolver,
    ForwardEulerCellSolver,
    LTGOSSolver,
    # Integrator
    get_parent_index,
    get_parent_value,
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
