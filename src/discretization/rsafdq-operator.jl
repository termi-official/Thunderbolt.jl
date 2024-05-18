# TODO try to reproduce this via the BlockOperator
struct AssembledRSAFDQ2022Operator{MatrixType <: BlockMatrix, ElementCacheType, FaceCacheType, TyingCacheType, DHType} <: AbstractBlockOperator
    J::MatrixType
    element_cache::ElementCacheType
    face_cache::FaceCacheType
    tying_cache::TyingCacheType
    dh::DHType
end

function AssembledRSAFDQ2022Operator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FaceQuadratureRuleCollection, tying::RSAFDQ2022TyingProblem)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.default_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    element_cache  = setup_element_cache(element_model, element_qr, ip, ip_geo)
    boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, ip, ip_geo)

    Jmech = create_sparsity_pattern(dh)
    @show size(Jmech)

    block_sizes = [ndofs(dh), length(tying.chambers)]
    total_size = sum(block_sizes)
    # First we define an empty dummy block array
    Jblock = BlockArray(spzeros(total_size,total_size), block_sizes, block_sizes)
    Jblock[Block(1,1)] = Jmech
    Jblock[Block(2,2)] = spzeros(1,1) # FIXME
    Jblock[Block(2,2)] .= 1.0

    AssembledRSAFDQ2022Operator(
        Jblock,
        element_cache,
        boundary_cache,
        EmptyTyingCache(), # FIXME
        dh,
    )
end

getJ(op::AssembledRSAFDQ2022Operator) = op.J
getJ(op::AssembledRSAFDQ2022Operator, i::Block) = @view op.J[i]

function update_linearization!(op::AssembledRSAFDQ2022Operator, u::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    Jdd = @view J[Block(1,1)]
    assembler = start_assemble(Jdd)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for cell in CellIterator(dh)
        dofs = celldofs(cell)
        fill!(Jₑ, 0)
        uₑ .= @view u[dofs]
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfaces(cell)
            assemble_face!(Jₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ)
    end

    #finish_assemble(assembler)
end

function update_linearization!(op::AssembledRSAFDQ2022Operator, u::AbstractVector, residual::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    Jdd = @view J[Block(1,1)]
    assembler = start_assemble(Jdd, residual)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @timeit_debug "loop" @inbounds for cell in CellIterator(dh)
        dofs = celldofs(cell)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ .= @view u[dofs]
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfaces(cell)
            assemble_face!(Jₑ, rₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
        @show dofs
        assemble!(assembler, dofs, Jₑ, rₑ)
    end

    #finish_assemble(assembler)
end
