############
# adapt.jl #
############
# Rules for this package's own values types. `CellValues` is Ferrite's, and its `FerriteKAExt`
# supplies the rule the device path needs -- a `CellValues` whose fields are device arrays, which is
# what the struct-of-arrays worker views are built from.
Adapt.@adapt_structure QuadratureValuesIterator
Adapt.@adapt_structure StaticQuadratureValues
