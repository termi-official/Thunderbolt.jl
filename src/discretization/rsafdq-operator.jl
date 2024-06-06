# TODO try to reproduce this via the BlockOperator
struct AssembledRSAFDQ2022Operator{MatrixType <: BlockMatrix, ElementCacheType, FaceCacheType, TyingCacheType, DHType} <: AbstractBlockOperator
    J::MatrixType
    element_cache::ElementCacheType
    face_cache::FaceCacheType
    tying_cache::TyingCacheType
    dh::DHType
end

function AssembledRSAFDQ2022Operator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FacetQuadratureRuleCollection, tying::RSAFDQ2022TyingInfo)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.geometric_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    element_cache  = setup_element_cache(element_model, element_qr, ip, ip_geo)
    boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, ip, ip_geo)

    Jmech = create_sparsity_pattern(dh)

    num_chambers = length(tying.chambers)
    block_sizes = [ndofs(dh), length(tying.chambers)]
    total_size = sum(block_sizes)
    # First we define an empty dummy block array
    Jblock = BlockArray(spzeros(total_size,total_size), block_sizes, block_sizes)
    Jblock[Block(1,1)] = Jmech
    Jblock[Block(2,2)] = spzeros(num_chambers,num_chambers)

    AssembledRSAFDQ2022Operator(
        Jblock,
        element_cache,
        boundary_cache,
        setup_tying_cache(tying, boundary_qr, ip, ip_geo),
        dh,
    )
end

getJ(op::AssembledRSAFDQ2022Operator) = op.J
getJ(op::AssembledRSAFDQ2022Operator, i::Block) = @view op.J[i]

function update_linearization!(op::AssembledRSAFDQ2022Operator, u::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    ud = @view u[Block(1)]
    up = @view u[Block(2)]

    Jdd = @view J[Block(1,1)]
    Jpd = @view J[Block(2,1)]
    Jdp = @view J[Block(1,2)]

    # Reset residual and Jacobian to 0
    assembler = start_assemble(Jdd)
    fill!(Jpd, 0.0)
    fill!(Jdp, 0.0)

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
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ)
    end

    # Assemble forward and backward coupling contributions
    for (chamber_index,chamber) ∈ enumerate(tying_cache.chambers)
        V⁰ᴰ = chamber.V⁰ᴰval
        chamber_pressure = u[chamber.pressure_dof_index_local] # We can also make this up[pressure_dof_index] with local index

        Jpd_current = @view Jpd[chamber_index,:]
        Jdp_current = @view Jdp[:,chamber_index]

        # We cannot update the residual for the displacement block here, because it would be assembled essentially twice.
        @timeit_debug "assemble forward coupler" assemble_LFSI_coupling_contribution_col!(Jdp_current, dh, ud, chamber_pressure, chamber)
        @timeit_debug "assemble backward coupler" assemble_LFSI_coupling_contribution_row!(Jpd_current, dh, ud, chamber_pressure, V⁰ᴰ, chamber)

        @info chamber_index, chamber_pressure, V⁰ᴰ
    end

    #finish_assemble(assembler)
end

function update_linearization!(op::AssembledRSAFDQ2022Operator, u::AbstractVector, residual::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    ud = @view u[Block(1)]
    up = @view u[Block(2)]

    residuald = @view residual[Block(1)]
    residualp = @view residual[Block(2)]

    Jdd = @view J[Block(1,1)]
    Jpd = @view J[Block(2,1)]
    Jdp = @view J[Block(1,2)]
    # Reset residual and Jacobian to 0
    assembler = start_assemble(Jdd, residuald)
    fill!(residualp, 0.0)
    fill!(Jpd, 0.0)
    fill!(Jdp, 0.0)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)

    @timeit_debug "loop" @inbounds for cell in CellIterator(dh)
        dofs = celldofs(cell)
        fill!(Jₑ, 0.0)
        fill!(rₑ, 0.0)
        uₑ .= @view u[dofs]
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, rₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end

    # Assemble forward and backward coupling contributions
    for (chamber_index,chamber) ∈ enumerate(tying_cache.chambers)
        V⁰ᴰ = chamber.V⁰ᴰval
        chamber_pressure = u[chamber.pressure_dof_index_local] # We can also make this up[pressure_dof_index] with local index

        Jpd_current = @view Jpd[chamber_index,:]
        Jdp_current = @view Jdp[:,chamber_index]

        # We cannot update the residual for the displacement block here, because it would be assembled essentially twice.
        @timeit_debug "assemble forward coupler" assemble_LFSI_coupling_contribution_col!(Jdp_current, dh, ud, chamber_pressure, chamber)
        @timeit_debug "assemble backward coupler" assemble_LFSI_coupling_contribution_row!(Jpd_current, residualp, dh, ud, chamber_pressure, V⁰ᴰ, chamber)

        @info "Chamber $chamber_index p=$chamber_pressure, V0=$V⁰ᴰ"
    end

    #finish_assemble(assembler)
end

function setup_operator(f::RSAFDQ20223DFunction, solver::AbstractNonlinearSolver)
    @unpack tying_info, structural_function = f
    # @unpack dh, constitutive_model, face_models, displacement_symbol = structural_function
    @unpack dh, constitutive_model, face_models = structural_function
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    displacement_symbol = first(dh.field_names)

    intorder = quadrature_order(structural_function, displacement_symbol)
    qr = QuadratureRuleCollection(intorder)
    qr_face = FacetQuadratureRuleCollection(intorder)

    return AssembledRSAFDQ2022Operator(
        dh, displacement_symbol, constitutive_model, qr, face_models, qr_face, tying_info
    )
end


BlockArrays.blocks(f::RSAFDQ20223DFunction) = (f.structural_function, f.tying_info)
