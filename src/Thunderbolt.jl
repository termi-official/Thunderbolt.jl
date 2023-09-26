module Thunderbolt

using Reexport, UnPack, StaticArrays
using OrderedCollections
@reexport using Ferrite

using JLD2

import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell
import Ferrite: vertices, edges, faces, sortedge, sortface

include("collections.jl")

include("mesh/meshes.jl")
include("mesh/coordinate_systems.jl")
include("mesh/tools.jl")
include("mesh/generators.jl")

include("modeling/microstructure.jl")

include("modeling/electrophysiology.jl")

include("modeling/mechanics/energies.jl")
include("modeling/mechanics/contraction.jl")
include("modeling/mechanics/active.jl")
include("modeling/mechanics/drivers.jl") # TODO better name. This is basically the quadrature point routine.

include("solver/solver.jl")

include("io.jl")

include("utils.jl")

include("exports.jl")

end
