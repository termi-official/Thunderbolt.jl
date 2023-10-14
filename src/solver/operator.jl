abstract type AbstractBilinearOperator end

struct AssembledBilinearOperator{MT,IT,CVT,SDH} <: AbstractBilinearOperator
    A::MT
    integrator::IT
    cv::CVT
    sdh::SDH
end

function update_operator!(bifo::AssembledBilinearOperator)
    @unpack A, cv, integrator, sdh  = bifo

    assembler = start_assemble(A)

    n_basefuncs = getnbasefunctions(cv)
    Aₑ = zeros(n_basefuncs, n_basefuncs)

    @inbounds for cell in CellIterator(sdh)
        fill!(Aₑ, 0)
        reinit!(cv, cell)
        assemble_element!(Aₑ, integrator, cv)
        assemble!(assembler, celldofs(cell), Aₑ)
    end
end

LinearAlgebra.mul!(out, bifo::AssembledBilinearOperator, in) = mul!(out, bifo.A, in)

struct BilinearMassForm
end

struct BilinearDiffusionForm{DTT}
    diffusion_tensor_field::DTT
end
