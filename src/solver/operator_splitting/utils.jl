# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? DataStructures.FasterForward : DataStructures.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = DataStructures.BinaryHeap{FT, ordering}(tstops)

    if isnothing(saveat)
        saveat = [t0, tf]
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        saveat = tf > t0 ? saveat : -saveat
        saveat = [t0:saveat:tf..., tf]
    else
        # We do not need to filter saveat like tstops because the saving
        # callback will ignore any times that are not between t0 and tf.
        saveat = collect(saveat)
    end
    saveat = DataStructures.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
end


need_sync(a::AbstractVector, b::AbstractVector) = true
need_sync(a::SubArray, b::AbstractVector)       = a.parent !== b
need_sync(a::AbstractVector, b::SubArray)       = a !== b.parent
need_sync(a::SubArray, b::SubArray)             = a.parent !== b.parent

function sync_vectors!(a, b)
    if need_sync(a, b) && a !== b
        a .= b
    end
end


# struct StandardSynchronizationMap{uMapType, pMapType}
#     unknown_indices::uMapType
#     parameter_indices::pMapType
# end

function forward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, dof_range, sync)
    forward_sync_internal!(outer_integrator, inner_integrator, dof_range)
    forward_sync_external!(outer_integrator, inner_integrator, sync)
end

function backward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync)
    backward_sync_internal!(outer_integrator, inner_integrator, sync)
end

# This is a bit tricky, because per default the operator splitting integrators share their solution vector. However, there is also the case
# when part of the problem is on a different device (thing e.g. about operator A being on CPU and B being on GPU).
# This case should be handled with special synchronizers.
forward_sync_internal!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::OperatorSplittingIntegrator, dof_range) = nothing
backward_sync_internal!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::OperatorSplittingIntegrator, dof_range) = nothing

function forward_sync_internal!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, unknown_indices)
    @views uouter = outer_integrator.u[unknown_indices]
    sync_vectors!(inner_integrator.uprev, uouter)
end
function backward_sync_internal!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, unknown_indices)
    @views uouter = outer_integrator.u[unknown_indices]
    sync_vectors!(uouter, inner_integrator.u)
end

# This is a noop, because operator splitting integrators do not have parameters
forward_sync_external!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::OperatorSplittingIntegrator, sync::NoExternalSynchronization) = nothing
function forward_sync_external!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync)
    synchronize_solution_with_parameters!(outer_integrator, inner_integrator.p, sync)
end

function synchronize_solution_with_parameters!(outer_integrator::OperatorSplittingIntegrator, p::Any, sync::StandardSynchronizationMap)
    error("Outer synchronizer not dispatched for parameter type $(typeof(p)).")
end

# If we encounter NullParameters, then we have the convention for the standard sync map that no external solution is necessary.
synchronize_solution_with_parameters!(outer_integrator::OperatorSplittingIntegrator, p::DiffEqBase.NullParameters, sync) = nothing

# Default convention is that the first parameter serves as a buffer for the external solution
# function synchronize_solution_with_parameters!(outer_integrator::OperatorSplittingIntegrator, p::Tuple, sync::StandardSynchronizationMap)
#     @views uouter = outer_integrator.u[sync.parameter_indices]
#     sync_vectors!(p[1], uouter)
# end
