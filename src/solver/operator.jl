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
        update_element_cache!(element_cache, cell, time)
        assemble_element!(Jₑ, uₑ, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        for local_face_index ∈ 1:nfaces(cell)
            face_is_initialized = false
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    if !face_is_initialized
                        face_is_initialized = true
                        reinit!(face_cache.fv, cell, local_face_index)
                    end
                    update_face_cache(cell, face_cache, time)

                    assemble_face!(Jₑ, uₑ, face_cache, time)
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
        update_element_cache!(element_cache, cell, time)
        assemble_element!(Jₑ, rₑ, uₑ, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        for local_face_index ∈ 1:nfaces(cell)
            face_is_initialized = false
            for face_cache ∈ face_caches
                if (cellid(cell), local_face_index) ∈ getfaceset(cell.grid, getboundaryname(face_cache))
                    if !face_is_initialized
                        face_is_initialized = true
                        reinit!(face_cache.fv, cell, local_face_index)
                    end
                    update_face_cache(cell, face_cache, time)

                    assemble_face!(Jₑ, rₑ, uₑ, face_cache, time)
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
        update_element_cache!(element_cache, cell, time)
        assemble_element!(Aₑ, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
    end

    #finish_assemble(assembler)
end

mul!(out, op::AssembledBilinearOperator, in) = mul!(out, op.A, in)
Base.eltype(op::AssembledBilinearOperator) = eltype(op.A)
Base.size(op::AssembledBilinearOperator, axis) = sisze(op.A, axis)
