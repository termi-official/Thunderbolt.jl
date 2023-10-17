abstract type AbstractNonlinearOperator end

struct AssembledNonlinearOperator{MatrixType, CacheType, DHType <: AbstractDofHandler} <: AbstractNonlinearOperator
    J::MatrixType
    element_cache::CacheType
    dh::DHType
end

function update_linearlization!(bifo::AssembledNonlinearOperator, u::Vector, time)
    @unpack J, element_cache, dh  = bifo

    assembler = start_assemble(A)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        update_element_cache!(element_cache, cell, time)
        assemble_element!(Jₑ, u, element_cache, time)
        assemble!(assembler, celldofs(cell), Jₑ)
    end

    #finish_assemble(assembler)
end

function update_linearlization!(bifo::AssembledNonlinearOperator, u::Vector, residual::Vector, time)
    @unpack J, element_cache, dh  = bifo

    assembler = start_assemble(J, residual)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        update_element_cache!(element_cache, cell, time)
        assemble_element!(Jₑ, rₑ, u, element_cache, time)
        assemble!(assembler, celldofs(cell), Jₑ, rₑ)
    end

    #finish_assemble(assembler)
end

"""
    mul!(out, bifo::AssembledNonlinearOperator, in)

Apply the action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out, bifo::AssembledNonlinearOperator, in) = mul!(out, bifo.J, in)

Base.eltype(biop::AssembledNonlinearOperator) = eltype(biop.A)
Base.size(biop::AssembledNonlinearOperator, axis) = sisze(biop.A, axis)


abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

struct AssembledBilinearOperator{MatrixType, CacheType, DHType <: AbstractDofHandler} <: AbstractBilinearOperator
    A::MatrixType
    element_cache::CacheType
    dh::DHType
end

function update_operator!(bifo::AssembledBilinearOperator, time)
    @unpack A, element_cache, dh  = bifo

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

mul!(out, biop::AssembledBilinearOperator, in) = mul!(out, biop.A, in)
Base.eltype(biop::AssembledBilinearOperator) = eltype(biop.A)
Base.size(biop::AssembledBilinearOperator, axis) = sisze(biop.A, axis)
