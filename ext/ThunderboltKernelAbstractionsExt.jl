module ThunderboltKernelAbstractionsExt

# The struct-of-arrays half of FerriteOperators' GPU device seam for Thunderbolt's element caches:
# `setup_device_instances` batches a cache over the workers once at setup, `device_worker_view`
# slices worker `w`'s out of that batch inside the assembly kernel. A cache author writes the pair,
# naming which fields batch and which are shared.
#
# The trigger stands for "FerriteOperators carries the device seam" -- the same condition that
# activates its own KernelAbstractions extension, and the version gate for `device_worker_view`,
# which FerriteOperators 0.4 does not have.

using Thunderbolt
using Ferrite

import Adapt: Adapt, adapt
import KernelAbstractions as KA

import FerriteOperators: AbstractGPUDevice, setup_device_instances, device_worker_view

import Thunderbolt:
    AnalyticalCoefficientElementCache,
    BilinearDiffusionElementCache,
    BilinearMassElementCache,
    SimpleMesh

import StaticArrays: SVector

# The coefficient caches the electrophysiology path reaches a device with are `isbits` -- constant
# coefficients, the conductivity-to-diffusivity quotient over them, and the Cartesian coordinate
# system, whose interpolation values are static matrices -- so they are shared read-only across
# workers. Only the values object carries per-cell state, and it batches.
setup_device_instances(device::AbstractGPUDevice, cache::BilinearMassElementCache, n::Int) =
    BilinearMassElementCache(cache.ρcache, setup_device_instances(device, cache.cellvalues, n))
device_worker_view(cache::BilinearMassElementCache, worker) =
    BilinearMassElementCache(cache.ρcache, device_worker_view(cache.cellvalues, worker))

setup_device_instances(device::AbstractGPUDevice, cache::BilinearDiffusionElementCache, n::Int) =
    BilinearDiffusionElementCache(cache.Dcache, setup_device_instances(device, cache.cellvalues, n))
device_worker_view(cache::BilinearDiffusionElementCache, worker) =
    BilinearDiffusionElementCache(cache.Dcache, device_worker_view(cache.cellvalues, worker))

# `nonzero_intervals` is read on the host, by `needs_update` off the *protocol* -- never by the
# element kernel. It still crosses the kernel boundary as part of the cache, so it becomes a
# statically sized vector, which is `isbits` and keeps the field's declared `AbstractVector{SVector{2,T}}`.
#
# The values object batches like any other even though this cache never `reinit!`s it: what the
# kernel reads out of it -- the reference shape values, the quadrature rule and the geometry mapping
# -- is the shared half of the batch, and recursing is the only route to a device-resident values
# object that does not name the backend.
function setup_device_instances(
    device::AbstractGPUDevice,
    cache::AnalyticalCoefficientElementCache,
    n::Int,
)
    return AnalyticalCoefficientElementCache(
        cache.cc,
        SVector{length(cache.nonzero_intervals)}(cache.nonzero_intervals),
        setup_device_instances(device, cache.cv, n),
    )
end
device_worker_view(cache::AnalyticalCoefficientElementCache, worker) =
    AnalyticalCoefficientElementCache(
        cache.cc,
        cache.nonzero_intervals,
        device_worker_view(cache.cv, worker),
    )

####################################
## Device handler over a SimpleMesh
####################################

"""
    Adapt.adapt_structure(backend::KA.Backend, dh::DofHandler{sdim, <:SimpleMesh})

The device-resident handler a device geometry cache is built from, for a handler whose grid is a
[`SimpleMesh`](@ref).

Ferrite's own rule builds a `FerriteKAExt.HostDofHandler`, whose grid field is declared
`G <: Ferrite.Grid{sdim}` -- so it takes no other `AbstractGrid`, and a `SimpleMesh` backed handler
is a `MethodError` there even though every accessor it uses forwards. Relaxing that bound to
`AbstractGrid{sdim}` upstream removes this method; until then the handler is rebuilt over the mesh's
own `Grid`, which carries the same cells and nodes in the same order.

The rebuild redistributes the dofs, so it asserts that it reproduced the numbering the operator's
matrix is indexed by. A device assembly against a handler that numbered differently would scatter
into the wrong rows, which no later check would catch.
"""
Adapt.adapt_structure(backend::KA.Backend, dh::DofHandler{sdim, <:SimpleMesh}) where {sdim} =
    adapt(backend, _grid_backed_handler(dh))

function _grid_backed_handler(dh::DofHandler{sdim, <:SimpleMesh}) where {sdim}
    rebuilt = DofHandler(dh.grid.grid)
    for sdh in dh.subdofhandlers
        sub = SubDofHandler(rebuilt, sdh.cellset)
        for (name, ip) in zip(sdh.field_names, sdh.field_interpolations)
            add!(sub, name, ip)
        end
    end
    close!(rebuilt)
    (
        ndofs(rebuilt) == ndofs(dh) &&
        rebuilt.cell_dofs == dh.cell_dofs &&
        rebuilt.cell_dofs_offset == dh.cell_dofs_offset
    ) || error(
        "Rebuilding the dof handler over the mesh's `Ferrite.Grid` -- which is what a device " *
        "handler has to be built from, see the method above -- did not reproduce the original dof " *
        "numbering, so the two do not address the same matrix. Assemble this operator on a CPU " *
        "device.",
    )
    return rebuilt
end

end
