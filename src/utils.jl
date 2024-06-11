
include("collections.jl")
include("quadrature_iterator.jl")

function celldofsview(dh::DofHandler, i::Int)
    if i == length(dh.cell_dofs_offset)
        return @views dh.cell_dofs[dh.cell_dofs_offset[i]:end]
    else
        return @views dh.cell_dofs[dh.cell_dofs_offset[i]:(dh.cell_dofs_offset[i+1]-1)]
    end
end


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
    orthogonalize_normal_system(v₁::Vec{dim,T}, v₂::Vec{dim,T})

Returns new vectors which are orthogonal to each other.
"""
@inline function orthogonalize_normal_system(v₁::Vec{2,T}, v₂::Vec{2,T}) where {T}
    w₁ = v₁
    w₂ = v₂ - (w₁ ⋅ v₂)*w₁
    return w₁, w₂
end

orthogonalize_system(v₁::Vec{2}, v₂::Vec{2}) = orthogonalize_normal_system(v₁/norm(v₁), v₂/norm(v₂))

"""
    orthogonalize_normal_system(v₁::Vec{3}, v₂::Vec{3}, v₃::Vec{3})

Returns new vectors which are orthogonal to each other.
"""
@inline function orthogonalize_normal_system(v₁::Vec{3}, v₂::Vec{3}, v₃::Vec{3})
    w₁ = v₁
    w₂ = v₂ - (w₁ ⋅ v₂)*w₁
    w₃ = v₃ - (w₁ ⋅ v₃)*w₁ - (w₂ ⋅ v₃)*w₂
    return w₁, w₂, w₃
end

orthogonalize_system(v₁::Vec{3}, v₂::Vec{3}, v₃::Vec{3}) = orthogonalize_normal_system(v₁/norm(v₁), v₂/norm(v₂), v₃/norm(v₃))

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

function mul!(y::AbstractVector{<:Number}, A_::ThreadedSparseMatrixCSR, x::AbstractVector{<:Number}, alpha::Number, beta::Number)
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

function mul!(y::AbstractVector{<:Number}, A_::ThreadedSparseMatrixCSR, x::AbstractVector{<:Number})
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


mtk_parameter_query_filter(discard_me, sym) = false
mtk_parameter_query_filter(param::ModelingToolkit.BasicSymbolic, sym) = true

function query_mtk_parameter_by_symbol(sys, sym::Symbol)
    symbol_list = ModelingToolkit.parameter_symbols(sys)
    idx = findfirst(param->mtk_parameter_query_filter(param,sym), symbol_list)
    idx === nothing && @error "Symbol $sym not found for system $sys."
    return symbol_list[idx]
end

"""
Examples:

* `DenseDataRange{Vector{Int}, Vector{Int}}` to map dofs (outer index) to elements (inner index)
* `DenseDataRange{Vector{Vec{3,Float64}}, Vector{Int}}` to store fluxes per quadrature point (inner index) per element (outer index)
"""
struct DenseDataRange{DataVectorType, IndexVectorType}
    data::DataVectorType
    offsets::IndexVectorType
end

@inline function get_data_for_index(r::DenseDataRange, i::Integer)
    i1 = r.offsets[i]
    i2 = r.offsets[i+1]-1
    return @view r.data[i1:i2]
end

struct ElementDofPair{IndexType}
    element_index::IndexType
    local_dof_index::IndexType
end

"""
"""
struct EAVector{T, EADataType <: AbstractVector{T}, IndexType <: AbstractVector{<:Int}, DofMapType <: AbstractVector{<:ElementDofPair}} <: AbstractVector{T}
    # Buffer for the per element data
    eadata::DenseDataRange{EADataType, IndexType}
    # Map from global dof index to element index and local dof index
    dof_to_element_map::DenseDataRange{DofMapType, IndexType}
end

@inline get_data_for_index(r::EAVector, i::Integer) = get_data_for_index(r.eadata, i)

function EAVector(dh::DofHandler)
    @assert length(dh.field_names) == 1
    map  = create_dof_to_element_map(dh)
    grid = get_grid(dh)
    # TODO subdomains
    num_entries = ndofs_per_cell(dh,1)*getncells(grid)
    eadata    = zeros(num_entries)
    eaoffsets = collect(1:ndofs_per_cell(dh,1):(num_entries+1))
    return EAVector(
        DenseDataRange(eadata, eaoffsets),
        map,
    )
end

# Transfer the element data into a vector
function ea_collapse!(b::Vector, bes::EAVector)
    ndofs = size(b, 1)
    @batch minbatch=ndofs÷Threads.nthreads() for dof ∈ 1:ndofs
        _ea_collapse_kernel!(b, dof, bes)
    end
end

@inline function _ea_collapse_kernel!(b::AbstractVector, dof::Integer, bes::EAVector)
    for edp ∈ get_data_for_index(bes.dof_to_element_map, dof)
        be_range = get_data_for_index(bes.eadata, edp.element_index)
        b[dof] += be_range[edp.local_dof_index]
    end
end

struct ChunkLocalAssemblyData{CellCacheType, ElementCacheType}
    cc::CellCacheType
    ec::ElementCacheType
end

create_dof_to_element_map(dh::DofHandler) = create_dof_to_element_map(Int, dh::DofHandler)

function create_dof_to_element_map(::Type{IndexType}, dh::DofHandler) where IndexType
    # Preallocate storage
    dof_to_element_vs = [Set{ElementDofPair{IndexType}}() for _ in 1:ndofs(dh)]
    # Fill set
    for cc in CellIterator(dh)
        eid = Ferrite.cellid(cc)
        for (ldi,dof) in enumerate(celldofs(cc))
            s = dof_to_element_vs[dof]
            push!(s, ElementDofPair(eid, ldi))
        end
    end
    #
    dof_to_element_vv = ElementDofPair{IndexType}[]
    offset = 1
    offsets = IndexType[]
    for dof in 1:ndofs(dh)
        append!(offsets, offset)
        s = dof_to_element_vs[dof]
        offset += length(s)
        append!(dof_to_element_vv, s)
    end
    append!(offsets, offset)
    #
    return DenseDataRange(
        dof_to_element_vv,
        offsets,
    )
end
