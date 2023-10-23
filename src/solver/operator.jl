abstract type AbstractNonlinearOperator end

struct AssembledNonlinearOperator{MatrixType, ElementCacheType, FaceCacheType, DHType <: AbstractDofHandler} <: AbstractNonlinearOperator
    J::MatrixType
    element_cache::ElementCacheType
    face_caches::FaceCacheType
    dh::DHType
end

function update_linearization!(op::AssembledNonlinearOperator, u::Vector, time)
    @unpack J, element_cache, face_caches, dh  = op

    assembler = start_assemble(A)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        uₑ = @view u[celldofs(cell)]
        # TODO instead of "cell" pass object with geometry information only
        assemble_element!(Jₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        for local_face_index ∈ 1:nfaces(cell)
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

    assembler = start_assemble(J, residual)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ = @view u[celldofs(cell)]
        # TODO instead of "cell" pass object with geometry information only
        assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        for local_face_index ∈ 1:nfaces(cell)
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    # TODO fix "(cell, local_face_index)" 
                    assemble_face!(Jₑ, rₑ, uₑ, (cell, local_face_index), face_cache, time)
                    break # only one integrator per face allowed!
                end
            end
        end
        assemble!(assembler, celldofs(cell), Jₑ, rₑ)
    end

    #finish_assemble(assembler)
end

"""
    mul!(out, op::AssembledNonlinearOperator, in)

Apply the action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out, op::AssembledNonlinearOperator, in) = mul!(out, op.J, in)

Base.eltype(op::AssembledNonlinearOperator) = eltype(op.A)
Base.size(op::AssembledNonlinearOperator, axis) = sisze(op.A, axis)


abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

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
        assemble_element!(Aₑ, cell, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
    end

    #finish_assemble(assembler)
end

mul!(out, op::AssembledBilinearOperator, in) = mul!(out, op.A, in)
Base.eltype(op::AssembledBilinearOperator) = eltype(op.A)
Base.size(op::AssembledBilinearOperator, axis) = sisze(op.A, axis)

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
        assemble_element!(bₑ, cell, element_cache, time)
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

function assemble_element!(bₑ, cell, element_cache::AnalyticalCoefficientElementCache, time)
    @unpack f, cv = element_cache
    reinit!(cv, cell)
    coords = getcoordinates(cell)
    for qp ∈ 1:getnquadpoints(cv)
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
