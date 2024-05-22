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

# function update_linearization!(op::AssembledRSAFDQ2022Operator, u::AbstractVector, time)
#     @unpack J, element_cache, face_cache, tying_cache, dh  = op

#     Jdd = @view J[Block(1,1)]
#     assembler = start_assemble(Jdd)

#     ndofs = ndofs_per_cell(dh)
#     Jₑ = zeros(ndofs, ndofs)
#     uₑ = zeros(ndofs)
#     uₜ = get_tying_dofs(tying_cache, u)
#     @inbounds for cell in CellIterator(dh)
#         dofs = celldofs(cell)
#         fill!(Jₑ, 0)
#         uₑ .= @view u[dofs]
#         @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
#         # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
#         @timeit_debug "assemble faces" for local_face_index ∈ 1:nfaces(cell)
#             assemble_face!(Jₑ, uₑ, cell, local_face_index, face_cache, time)
#         end
#         @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
#         assemble!(assembler, dofs, Jₑ)
#     end

#     #finish_assemble(assembler)
# end

function update_linearization!(op::AssembledRSAFDQ2022Operator, u::AbstractVector, residual::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    ud = u[Block(1)]
    up = u[Block(2)]

    residuald = @view residual[Block(1)]
    residualp = @view residual[Block(2)]

    Jdd = @view J[Block(1,1)]
    Jpd = @view J[Block(2,1)]
    Jdp = @view J[Block(1,2)]
    # Reset residual and Jacobian to 0
    assembler = start_assemble(Jdd, residuald)
    fill!(residualp, 0.0)
    fill!(Jdp, 0.0)
    fill!(Jdp, 0.0)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)

    rₑdebug = zeros(ndofs)
    Jₑdebug = zeros(ndofs, ndofs)
    Jₑdebug2 = zeros(ndofs, ndofs)
    h = 1e-12

    @timeit_debug "loop" @inbounds for cell in CellIterator(dh)
        dofs = celldofs(cell)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ .= @view ud[dofs]
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfaces(cell)
            assemble_face!(Jₑ, rₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ, rₑ)

        # Debug checks
        fill!(rₑ, 0.0)
        fill!(Jₑdebug, 0.0)
        fill!(Jₑdebug2, 0.0)
        # Computed Jacobian
        assemble_tying!(Jₑdebug, rₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble_element!(Jₑdebug, rₑ, uₑ, cell, element_cache, time)
        # Finite differences
        for i in 1:length(uₑ)
            fill!(rₑdebug, 0.0)
            direction = zeros(length(uₑ))
            direction[i] = h
            assemble_tying!(Jₑ, rₑdebug, uₑ .+ direction, uₜ, cell, tying_cache, time)
            assemble_element!(Jₑ, rₑdebug, uₑ .+ direction, cell, element_cache, time)
            Jₑdebug2[:, i] .= (rₑdebug .- rₑ) / h
        end
        @assert all(isapprox.(Jₑdebug, Jₑdebug2, atol=norm(Jₑdebug, 1.0)/length(Jₑdebug))) "$rₑdebug, $rₑ, $Jₑdebug, $Jₑdebug2"
    end

    # Assemble forward and backward coupling contributions
    for (chamber_index,chamber) ∈ enumerate(tying_cache.chambers)
        V⁰ᴰ = chamber.V⁰ᴰval
        @show chamber_pressure = u[chamber.pressure_dof_index] # We can also make this up[pressure_dof_index] with local index

        Jpd_current = @view Jpd[chamber_index,:]
        Jdp_current = @view Jdp[:,chamber_index]

        # We cannot pass the residual for the displacement block here, because it would be assembled essentially twice.
        # @timeit_debug "assemble forward coupler" assemble_LFSI_coupling_contribution_col!(Jdp_current, residuald dh, ud, chamber_pressure, chamber)
        @timeit_debug "assemble forward coupler" assemble_LFSI_coupling_contribution_col!(Jdp_current, dh, ud, chamber_pressure, chamber)
        @timeit_debug "assemble backward coupler" assemble_LFSI_coupling_contribution_row!(Jpd_current, residualp, dh, ud, chamber_pressure, V⁰ᴰ, chamber)
        # dV = 0.0
        # for i in 1:length(Jpd_current)
        #     dV += Jpd_current[i]*u[i]
        # end
        # @show dV
        # J[chamber.pressure_dof_index,chamber.pressure_dof_index] = 0.1
        # residual[chamber.pressure_dof_index] = 0.0#0.1*(chamber_pressure-time/100+0.5)

        Jpd_debug = copy(Jpd_current)
        fill!(Jpd_debug, 0.0)
        Jdp_debug = copy(Jdp_current)
        fill!(Jdp_debug, 0.0)
        Jpd_debug2 = copy(Jpd_current)
        fill!(Jpd_debug2, 0.0)
        Jdp_debug2 = copy(Jdp_current)
        fill!(Jdp_debug2, 0.0)
        Jpd_discard = copy(Jpd_current)
        fill!(Jpd_discard, 0.0)
        Jdp_discard = copy(Jdp_current)
        fill!(Jdp_discard, 0.0)
        rddebug = copy(residuald)
        fill!(rddebug, 0.0)
        rpdebug = copy(residualp)
        fill!(rpdebug, 0.0)
        rddebug2 = copy(residuald)
        fill!(rddebug2, 0.0)
        rpdebug2 = copy(residualp)
        fill!(rpdebug2, 0.0)

        # Computed stuff
        assemble_LFSI_coupling_contribution_col!(Jdp_debug, rddebug, dh, ud, chamber_pressure, chamber)
        assemble_LFSI_coupling_contribution_row!(Jpd_debug, rpdebug, dh, ud, chamber_pressure, V⁰ᴰ, chamber)

        # Finite difference for the forward contribution (w.r.t. pressure)
        assemble_LFSI_coupling_contribution_col!(Jdp_discard, rddebug2, dh, ud, chamber_pressure+h, chamber)
        Jdp_debug2 .= (rddebug2 .- rddebug) / h
        @assert all(isapprox.(Jdp_debug, Jdp_debug2, atol=norm(Jdp_debug, 1.0)/length(Jdp_debug))) "$Jdp_debug, $Jdp_debug2"

        Jpd_discard = copy(Jpd_current)
        fill!(Jpd_discard, 0.0)
        # Finite difference for the back contribution (w.r.t. displacement)
        direction = zeros(length(ud))
        for i in 1:length(ud)
            direction[i] = h
            fill!(rpdebug2, 0.0)
            assemble_LFSI_coupling_contribution_row!(Jpd_discard, rpdebug2, dh, ud .+ direction, chamber_pressure, V⁰ᴰ, chamber)
            Jpd_debug2[i, :] .= (rpdebug2 .- rpdebug) / h
            direction[i] = 0.0
        end
        @assert all(isapprox.(Jpd_debug2, Jpd_debug, atol=norm(Jpd_debug, 1.0)/length(Jpd_debug))) "$(Vector(Jpd_debug)), $(Vector(Jpd_debug2))"

    end

    #finish_assemble(assembler)
end