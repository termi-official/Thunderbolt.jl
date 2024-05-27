# Some dispatches to make the dispatcher happy
*(::Thunderbolt.ThreadedSparseMatrixCSR, ::ModelingToolkit.Symbolics.Arr{<:Any, 1}) = @error "Not implemented"
mul!(::Thunderbolt.ModelingToolkit.JumpProcesses.ExtendedJumpArray, ::Thunderbolt.ThreadedSparseMatrixCSR, ::AbstractVector{<:Number})= @error "Not implemented"
