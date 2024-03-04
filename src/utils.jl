
include("collections.jl")
include("quadrature_iterator.jl")

function calculate_element_volume(cell, cellvalues_u, uₑ)
    reinit!(cellvalues_u, cell)
    evol::Float64=0.0;
    @inbounds for qp in QuadratureIterator(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        ∇u = function_gradient(cellvalues_u, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        evol += J * dΩ
    end
    return evol
end;

function calculate_volume_deformed_mesh(w, dh::DofHandler, cellvalues_u)
    evol::Float64 = 0.0;
    @inbounds for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        nu = getnbasefunctions(cellvalues_u)
        global_dofs_u = global_dofs[1:nu]
        uₑ = w[global_dofs_u]
        δevol = calculate_element_volume(cell, cellvalues_u, uₑ)
        evol += δevol;
    end
    return evol
end;

@inline angle(v1::Vec{dim,T}, v2::Vec{dim,T}) where {dim, T} = acos((v1 ⋅ v2)/(norm(v1)*norm(v2)))
@inline angle_deg(v1::Vec{dim,T}, v2::Vec{dim,T}) where {dim, T} =  rad2deg(angle(v1, v2))

"""
    unproject(v::Vec{dim,T}, n::Vec{dim,T}, α::T)::Vec{dim, T}

Unproject the vector `v` from the plane with normal `n` such that the angle between `v` and the
resulting vector is `α` (given in radians).

!!! note It is assumed that the vectors are normalized and orthogonal, i.e. `||v|| = 1`, `||n|| = 1`
         and `v \\cdot n = 0`.
"""
@inline function unproject(v::Vec{dim,T}, n::Vec{dim,T}, α::T)::Vec{dim, T} where {dim, T}
    @debug @assert norm(v) ≈ 1.0
    @debug @assert norm(n) ≈ 1.0
    @debug @assert v ⋅ n ≈ 0.0

    α ≈ π/2.0 && return n # special case to prevent division by 0

    λ = (sqrt(1-cos(α)^2))/cos(α)
    return v + λ * n
end

"""
    rotate_around(v::Vec{dim,T}, a::Vec{dim,T}, θ::T)::Vec{dim,T}

Perform a Rodrigues' rotation of the vector `v` around the axis `a` with `θ` radians.

!!! note It is assumed that the vectors are normalized, i.e. `||v|| = 1` and `||a|| = 1`.
"""
@inline function rotate_around(v::Vec{dim,T}, a::Vec{dim,T}, θ::T)::Vec{dim,T} where {dim, T}
    @debug @assert norm(n) ≈ 1.0

    return v * cos(θ) + (a × v) * sin(θ) + a * (a ⋅ v) * (1-cos(θ))
end

"""
    orthogonalize(v₁::Vec{dim,T}, v₂::Vec{dim,T})::Vec{dim,T}

Returns a new `v₁` which is orthogonal to `v₂`.
"""
@inline function orthogonalize(v₁::Vec{dim,T}, v₂::Vec{dim,T})::Vec{dim,T} where {dim, T} 
    return v₁ - (v₁ ⋅ v₂)*v₂
end

"""
"""
function generate_nodal_quadrature_rule(ip::Interpolation{ref_shape, order}) where {ref_shape, order}
    n_base = Ferrite.getnbasefunctions(ip)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape, Float64}(ones(length(positions)), positions)
end

"""
    ThreadedSparseMatrixCSR
Threaded version of SparseMatrixCSR.

Based on https://github.com/BacAmorim/ThreadedSparseCSR.jl .
"""
struct ThreadedSparseMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    A::SparseMatrixCSR{1,Tv,Ti}
end

function ThreadedSparseMatrixCSR(m::Integer, n::Integer, rowptr::Vector{Ti}, colval::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
    ThreadedSparseMatrixCSR(SparseMatrixCSR{1}(m,n,rowptr,colval,nzval))
end

function ThreadedSparseMatrixCSR(a::Transpose{Tv,<:SparseMatrixCSC} where Tv)
    ThreadedSparseMatrixCSR(SparseMatrixCSR(a))
end

function mul!(y::AbstractVector, A_::ThreadedSparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    A = A_.A
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())
    
    @batch minbatch = size(y, 1) ÷ Threads.nthreads() for row in 1:size(y, 1)
        @inbounds begin
            v = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz]
                v += A.nzval[nz]*x[col]
            end
            y[row] = alpha*v + beta*y[row]
        end
    end

    return y
end

function mul!(y::AbstractVector, A_::ThreadedSparseMatrixCSR, x::AbstractVector)
    A = A_.A
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    @batch minbatch = size(y, 1) ÷ Threads.nthreads() for row in 1:size(y, 1)
        @inbounds begin
            v = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz]
                v += A.nzval[nz]*x[col]
            end
            y[row] = v
        end
    end

    return y
end

function mul(A::ThreadedSparseMatrixCSR, x::AbstractVector)
    y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
    return mul!(y, A, x)
end
*(A::ThreadedSparseMatrixCSR, v::AbstractVector) = mul(A,v)
*(A::ThreadedSparseMatrixCSR, v::BlockArrays.FillArrays.AbstractZeros{<:Any, 1}) = mul(A,v)
*(A::ThreadedSparseMatrixCSR, v::BlockArrays.ArrayLayouts.LayoutVector) = mul(A,v)

Base.eltype(A::ThreadedSparseMatrixCSR) = Base.eltype(A.A)
getrowptr(A::ThreadedSparseMatrixCSR) = getrowptr(A.A)
getnzval(A::ThreadedSparseMatrixCSR) = getnzval(A.A)
getcolval(A::ThreadedSparseMatrixCSR) = getnzval(A.A)
issparse(A::ThreadedSparseMatrixCSR) = issparse(A.A)
nnz(A::ThreadedSparseMatrixCSR) = nnz(A.A)
nonzeros(A::ThreadedSparseMatrixCSR) = nonzeros(A.A)
Base.size(A::ThreadedSparseMatrixCSR) = Base.size(A.A)
Base.size(A::ThreadedSparseMatrixCSR,i) = Base.size(A.A,i)
IndexStyle(::Type{<:ThreadedSparseMatrixCSR}) = IndexCartesian()

# Internal helper to throw uniform error messages on problems with multiple subdomains
@noinline check_subdomains(dh::Ferrite.AbstractDofHandler) = length(dh.subdofhandlers) == 1 || throw(ArgumentError("Using DofHandler with multiple subdomains is not currently supported"))
@noinline check_subdomains(grid::Ferrite.AbstractGrid) = length(elementtypes(grid)) == 1 || throw(ArgumentError("Using mixed grid is not currently supported"))

@inline function quadrature_order(problem, fieldname)
    @unpack dh = problem
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the quadrature order determination."
    2*Ferrite.getorder(Ferrite.getfieldinterpolation(dh.subdofhandlers[1], fieldname))
end
