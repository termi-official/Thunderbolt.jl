module OS

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import SciMLBase, DiffEqBase, DataStructures

import OrdinaryDiffEqCore

import UnPack: @unpack
import DiffEqBase: init, TimeChoiceIterator

abstract type AbstractOperatorSplitFunction <: DiffEqBase.AbstractODEFunction{true} end
abstract type AbstractOperatorSplittingAlgorithm end
abstract type AbstractOperatorSplittingCache end

include("operator_splitting/function.jl")
include("operator_splitting/problem.jl")
include("operator_splitting/integrator.jl")
include("operator_splitting/solver.jl")
include("operator_splitting/utils.jl")

export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov,
    DiffEqBase, init, TimeChoiceIterator,
    NoExternalSynchronization

end
