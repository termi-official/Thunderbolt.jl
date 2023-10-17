# struct JacobiPreconditioner{T}
#     Ainv::Diagonal{T}
# end
# function JacobiPreconditioner(A)
#     JacobiPreconditioner{eltype(A)}(inv(Diagonal(diag(A))))
# end
# LinearAlgebra.ldiv!(P::JacobiPreconditioner{T}, b) where {T} = mul!(b, P.Ainv, b)
# LinearAlgebra.ldiv!(y, P::JacobiPreconditioner{T}, b) where {T} = mul!(y, P.Ainv, b)
# import Base: \
# function (\)(P::JacobiPreconditioner{T}, b) where {T}
#     return ldiv!(similar(b), P.Ainv, b)
# end

# using Polyester
# struct JacobiPreconditioner{T}
#     Ainv::Vector{T}
# end
# function JacobiPreconditioner(A)
#     JacobiPreconditioner{eltype(A)}(inv.(Vector(diag(A))))
# end
# function LinearAlgebra.ldiv!(y::TV, P::JacobiPreconditioner{T}, b::TV) where {T, TV<:AbstractVector}
#     @batch minbatch = size(y, 1) ÷ Threads.nthreads() for row in 1:length(P.Ainv)
#         @inbounds begin
#             y[row] = P.Ainv[row]*b[row]
#         end
#     end
# end
