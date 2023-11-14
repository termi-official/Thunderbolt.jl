# TODO split nonlinear operator and the linearization concepts
# TODO energy based operator?
# TODO maybe a trait system for operators?
"""
    AbstractNonlinearOperator

Models of a nonlinear function F(u).

Interface:
    (op::AbstractNonlinearOperator)(residual::AbstractVector, in::AbstractNonlinearOperator)
    eltype()
    size()

    # linearization
    mul!(out, op::AbstractNonlinearOperator, in)
    mul!(out, op::AbstractNonlinearOperator, in, α, β)
    update_linearization!(op::AbstractNonlinearOperator, u::AbstractVector, time)
    update_linearization!(op::AbstractNonlinearOperator, u::AbstractVector, residual::AbstractVector, time)
"""
abstract type AbstractNonlinearOperator end

getJ(op) = error("J is not explicitly accessible for given operator")

function *(op::AbstractNonlinearOperator, x::AbstractVector)
    y = similar(x)
    mul!(y, op, x)
    return y
end

# TODO constructor which checks for axis compat
struct BlockOperator{OPS <: Tuple}
    # TODO maybe SMatrix?
    operators::OPS # stored row by row as in [1 2; 3 4]
end

getJ(op::BlockOperator) = mortar(reshape([getJ(opi) for opi ∈ op.operators], (isqrt(length(op.operators)), isqrt(length(op.operators)))))

function *(op::BlockOperator, x::AbstractVector)
    y = similar(x)
    mul!(y, op, x)
    return y
end

# TODO optimize
mul!(y, op::BlockOperator, x) = mul!(y, getJ(op), x)

# TODO can we be clever with broadcasting here?
function update_linearization!(op::BlockOperator, u::BlockVector, time)
    @warn "linearization not functional for actually coupled problems!" maxlog=1
    for opi ∈ op.operators
        update_linearization!(opi, u, time)
    end
end

# TODO can we be clever with broadcasting here?
function update_linearization!(op::BlockOperator, u::BlockVector, residual::BlockVector, time)
    @warn "linearization not functional for actually coupled problems!" maxlog=1
    nops = length(op.operators)
    nrows = isqrt(nops)
    for i ∈ 1:nops
        i1 = Block(div(i-1, nrows) + 1) # index shift due to 1-based indices
        row_residual = @view residual[i1]
        u_ = @view u[Block(rem(i-1, nrows) + 1)] # TODO REMOVEME
        @timeit_debug "update block $i1" update_linearization!(op.operators[i], u_, row_residual, time)
    end
end

# TODO can we be clever with broadcasting here?
function mul!(out::BlockVector, op::BlockOperator, in::BlockVector)
    out .= 0.0
    # 5-arg-mul over 3-ar-gmul because the bocks would overwrite the solution!
    mul!(out, op, in, 1.0, 1.0)
end

# TODO can we be clever with broadcasting here?
function mul!(out::BlockVector, op::BlockOperator, in::BlockVector, α, β)
    nops = length(op.operators)
    nrows = isqrt(nops)
    for i ∈ 1:nops
        i1, i2 = Block.(divrem(i-1, nrows) .+1) # index shift due to 1-based indices
        in_next  = @view in[i1] 
        out_next = @view out[i2]
        mul!(out_next, op.operators[i], in_next, α, β)
    end
end

struct AssembledNonlinearOperator{MatrixType, ElementCacheType, FaceCacheType, DHType <: AbstractDofHandler} <: AbstractNonlinearOperator
    J::MatrixType
    element_cache::ElementCacheType
    face_caches::FaceCacheType
    dh::DHType
end

getJ(op::AssembledNonlinearOperator) = op.J

function update_linearization!(op::AssembledNonlinearOperator, u::Vector, time)
    @unpack J, element_cache, face_caches, dh  = op

    assembler = start_assemble(J)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        uₑ = @view u[celldofs(cell)]
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfaces(cell)
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    # TODO fix "(cell, local_face_index)" 
                    assemble_face!(Jₑ, uₑ, (cell, local_face_index), face_cache, time)
                    break # only one integrator per face allowed!
                end
            end
        end
        assemble!(assembler, celldofs(cell), Jₑ)
    end

    #finish_assemble(assembler)
end

function update_linearization!(op::AssembledNonlinearOperator, u::Vector, residual::Vector, time)
    @unpack J, element_cache, face_caches, dh  = op

    assembler = start_assemble(J)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ = @view u[celldofs(cell)]
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfaces(cell)
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    # TODO fix "(cell, local_face_index)" 
                    assemble_face!(Jₑ, rₑ, uₑ, (cell, local_face_index), face_cache, time)
                    break # only one integrator per face allowed!
                end
            end
        end
        assemble!(assembler, celldofs(cell), Jₑ)
        residual[celldofs(cell)] += rₑ # separate because the residual might contain more external stuff
    end

    #finish_assemble(assembler)
end

"""
    mul!(out, op::AssembledNonlinearOperator, in)
    mul!(out, op::AssembledNonlinearOperator, in, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out, op::AssembledNonlinearOperator, in) = mul!(out, op.J, in)
mul!(out, op::AssembledNonlinearOperator, in, α, β) = mul!(out, op.J, in, α, β)

Base.eltype(op::AssembledNonlinearOperator) = eltype(op.A)
Base.size(op::AssembledNonlinearOperator, axis) = sisze(op.A, axis)


abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

update_linearization!(op::AbstractBilinearOperator, u, residual, time) = nothing # TODO REMOVEME
update_linearization!(op::AbstractBilinearOperator, u, time) = nothing # TODO REMOVEME

struct AssembledBilinearOperator{MatrixType, CacheType, DHType <: AbstractDofHandler} <: AbstractBilinearOperator
    A::MatrixType
    element_cache::CacheType
    dh::DHType
end

function update_operator!(op::AssembledBilinearOperator, time)
    @unpack A, element_cache, dh  = op

    assembler = start_assemble(A)

    ndofs = ndofs_per_cell(dh)
    Aₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Aₑ, 0)
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
    end

    #finish_assemble(assembler)
end

mul!(out, op::AssembledBilinearOperator, in) = mul!(out, op.A, in)
mul!(out, op::AssembledBilinearOperator, in, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::AssembledBilinearOperator) = eltype(op.A)
Base.size(op::AssembledBilinearOperator, axis) = sisze(op.A, axis)

"""
    DiagonalOperator <: AbstractBilinearOperator

Literally a "diagonal matrix".
"""
struct DiagonalOperator{TV <: AbstractVector} <: AbstractBilinearOperator
    values::TV
end

mul!(out, op::DiagonalOperator, in) = out .= op.values .* out
mul!(out, op::DiagonalOperator, in, α, β) = out .= α * op.values .* in + β * out
Base.eltype(op::DiagonalOperator) = eltype(op.values)
Base.size(op::DiagonalOperator, axis) = length(op.values)

getJ(op::DiagonalOperator) = spdiagm(op.values)

"""
    NullOperator <: AbstractBilinearOperator

Literally a "null matrix".
"""

struct NullOperator{T, SIN, SOUT} <: AbstractBilinearOperator
end

mul!(out, op::NullOperator, in) = out .= 0.0
mul!(out, op::NullOperator, in, α, β) = out .= β*out
Base.eltype(op::NullOperator{T}) where {T} = T
Base.size(op::NullOperator{T,S1,S2}, axis) where {T,S1,S2} = axis == 1 ? S1 : (axis == 2 ? S2 : error("faulty axis!"))

getJ(op::NullOperator{T, SIN, SOUT}) where {T, SIN, SOUT} = spzeros(T,SIN,SOUT)

###############################################################################
abstract type AbstractLinearOperator end

"""
    LinearNullOperator <: AbstractLinearOperator

Literally the null vector.
"""
struct LinearNullOperator{T,S} <: AbstractLinearOperator
end
Base.eltype(op::LinearNullOperator{T,S}) where {T,S} = T
Base.size(op::LinearNullOperator{T,S}) where {T,S} = S

update_operator!(op::LinearNullOperator, time) = nothing
add!(b::Vector, op::LinearNullOperator) = nothing
needs_update(op::LinearNullOperator, t) = false


struct LinearOperator{VectorType, CacheType, DHType <: AbstractDofHandler} <: AbstractLinearOperator
    b::VectorType
    element_cache::CacheType
    dh::DHType
end

function update_operator!(op::LinearOperator, time)
    @unpack b, element_cache, dh  = op

    # assembler = start_assemble(b)

    ndofs = ndofs_per_cell(dh)
    bₑ = zeros(ndofs)
    fill!(b, 0.0)
    @inbounds for cell in CellIterator(dh)
        fill!(bₑ, 0)
        @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
        # assemble!(assembler, celldofs(cell), bₑ)
        b[celldofs(cell)] .+= bₑ
    end

    #finish_assemble(assembler)
end

add!(b::Vector, op::LinearOperator) = b .+= op.b
Base.eltype(op::LinearOperator) = eltype(op.b)
Base.size(op::LinearOperator) = sisze(op.b)

# TODO where to put these?
create_linear_operator(dh, ::NoStimulationProtocol) = LinearNullOperator{Float64, ndofs(dh)}()
function create_linear_operator(dh, protocol::AnalyticalTransmembraneStimulationProtocol)
    ip = dh.subdofhandlers[1].field_interpolations[1]
    ip_g = Ferrite.default_interpolation(typeof(getcells(Ferrite.get_grid(dh), 1)))
    qr = QuadratureRule{Ferrite.getrefshape(ip_g)}(Ferrite.getorder(ip_g)+1)
    cv = CellValues(qr, ip, ip_g) # TODO replace with something more lightweight
    return LinearOperator(
        zeros(ndofs(dh)),
        AnalyticalCoefficientElementCache(
            protocol.f,
            protocol.nonzero_intervals,
            cv
        ),
        dh
    )
end
struct AnalyticalCoefficientElementCache{F <: AnalyticalCoefficient, T, CV}
    f::F
    nonzero_intervals::Vector{SVector{2,T}}
    cv::CV
end

@inline function _reinit_tb!(cv, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    n_func_basefuncs = Ferrite.getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in pairs(Ferrite.getweights(cv.qr))
        fecv_J = zero(Tensor{2,dim,T})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        # detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        # Jinv = inv(fecv_J)
        # for j in 1:n_func_basefuncs
        #     # cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
        #     cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        # end
    end
end
function assemble_element!(bₑ, cell, element_cache::AnalyticalCoefficientElementCache, time)
    @unpack f, cv = element_cache
    coords = getcoordinates(cell)
    #reinit!(cv, coords) # TODO reinit geometry values only...
    _reinit_tb!(cv, coords)
    
    @inbounds for qp ∈ 1:getnquadpoints(cv)
        x = spatial_coordinate(cv, qp, coords)
        fx = f.f(x,time)
        dΩ = getdetJdV(cv, qp)
        for j ∈ 1:getnbasefunctions(cv)
            δu = shape_value(cv, qp, j)
            bₑ[j] += fx * δu * dΩ
        end
    end
end

function needs_update(op::LinearOperator{<:Any, <: AnalyticalCoefficientElementCache}, t)
    for nonzero_interval ∈ op.element_cache.nonzero_intervals
        nonzero_interval[1] ≤ t ≤ nonzero_interval[2] && return true
    end
    return false
end
