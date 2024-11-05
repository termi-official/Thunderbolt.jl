# struct StandardAssemblyInfo{T}
#     time::T
# end

# allocate_Je(sdh::SubDofHandler, info::StandardAssemblyInfo) = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
# allocate_ue(sdh::SubDofHandler, info::StandardAssemblyInfo) = zeros(ndofs_per_cell(sdh))
# allocate_re(sdh::SubDofHandler, info::StandardAssemblyInfo) = zeros(ndofs_per_cell(sdh))
# query_ue!(ue, u, cell, info::StandardAssemblyInfo)          = uₑ .= @view u[celldofs(cell)]
# uncondense_ue!(u, uₑ, cell, info::StandardAssemblyInfo)     = nothing # Standard assembly does not support condensation

allocate_Je(sdh::SubDofHandler, info) = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
allocate_ue(sdh::SubDofHandler, info) = zeros(ndofs_per_cell(sdh))
allocate_re(sdh::SubDofHandler, info) = zeros(ndofs_per_cell(sdh))
query_ue!(uₑ, u, cell, info)          = uₑ .= @view u[celldofs(cell)]
uncondense_ue!(u, uₑ, cell, info)     = nothing # Standard assembly does not support condensation

# query_time(info::StandardAssemblyInfo) = info.time

"""
    AssembledNonlinearOperator2([matrix_type,] dh, ...)

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
    assemble_face! -> update jacobian/residual contribution for boundary

TODO
    assemble_interface! -> update jacobian/residual contribution for interface contributions (e.g. DG or FSI)
"""
struct AssembledNonlinearOperator2{MatrixType <: AbstractSparseMatrix, ElementModelType, FacetModelType, DHType <: AbstractDofHandler} <: AbstractNonlinearOperator
    # Concrete type for the linearization
    J::MatrixType
    # Volumetric models and quadrature rule
    element_model::ElementModelType
    element_qrc::Union{<:QuadratureRuleCollection, Nothing}
    # Facet models and quadrature rule
    face_model::FacetModelType
    face_qrc::Union{<:FacetQuadratureRuleCollection, Nothing}
    dh::DHType
end

function AssembledNonlinearOperator2(dh::AbstractDofHandler, element_model, element_qrc::QuadratureRuleCollection)
    AssembledNonlinearOperator2(
        allocate_matrix(dh),
        element_model, element_qrc,
        nothing, nothing,
        dh,
    )
end

#Utility constructor to get the nonlinear operator for a single field problem.
function AssembledNonlinearOperator2(dh::AbstractDofHandler, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FacetQuadratureRuleCollection)
    AssembledNonlinearOperator2(
        allocate_matrix(dh),
        element_model, element_qrc,
        boundary_model, boundary_qrc,
        dh,
    )
end

getJ(op::AssembledNonlinearOperator2) = op.J

function update_linearization!(op::AssembledNonlinearOperator2, u::AbstractVector, assembly_info)
    @unpack J, dh  = op
    @unpack element_model, element_qrc = op
    @unpack face_model, face_qrc = op

    assembler = start_assemble(J)
    for sdh in dh.subdofhandlers
        # Prepare evaluation caches
        element_qr  = getquadraturerule(element_qrc, sdh)
        face_qr     = face_model === nothing ? nothing : getquadraturerule(face_qrc, sdh)

        # Build evaluation caches
        element_cache  = setup_element_cache(element_model, element_qr, sdh)
        face_cache     = setup_boundary_cache(face_model, face_qr, sdh)

        # Function barrier
        _update_linearization_on_subdomain_J!(assembler, sdh, element_cache, face_cache, u, assembly_info)
    end
    # finish_assemble(assembler)
end

function _update_linearization_on_subdomain_J!(assembler, sdh, element_cache, face_cache, u, assembly_info)
    # Allocate buffers
    Jₑ = allocate_Je(sdh, assembly_info)
    uₑ = allocate_ue(sdh, assembly_info)
    @inbounds for cell in CellIterator(sdh)
        # Prepare buffers for current iteration
        fill!(Jₑ, 0)
        query_ue!(uₑ, u, cell, assembly_info)

        # Fill buffers
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, assembly_info)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, uₑ, cell, local_face_index, face_cache, assembly_info)
        end
        assemble!(assembler, celldofs(cell), Jₑ)

        # The assembly procedure might include local solves which have updated the local solution vector
        uncondense_ue!(u, uₑ, cell, assembly_info)
    end
end

function update_linearization!(op::AssembledNonlinearOperator2, residual::AbstractVector, u::AbstractVector, assembly_info)
    @unpack J, dh  = op
    @unpack element_model, element_qrc = op
    @unpack face_model, face_qrc = op

    assembler = start_assemble(J, residual)
    for sdh in dh.subdofhandlers
        # Prepare evaluation caches
        element_qr  = getquadraturerule(element_qrc, sdh)
        face_qr     = face_model === nothing ? nothing : getquadraturerule(face_qrc, sdh)

        # Build evaluation caches
        element_cache  = setup_element_cache(element_model, element_qr, sdh)
        face_cache     = setup_boundary_cache(face_model, face_qr, sdh)

        # Function barrier
        _update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, face_cache, u, assembly_info)
    end
    # finish_assemble(assembler)
end

function _update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, face_cache, u, assembly_info)
    # Allocate buffers
    Jₑ = allocate_Je(sdh, assembly_info)
    uₑ = allocate_ue(sdh, assembly_info)
    rₑ = allocate_re(sdh, assembly_info)

    @inbounds for cell in CellIterator(sdh)
        # Prepare buffers for current iteration
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        query_ue!(uₑ, u, cell, assembly_info)

        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, assembly_info)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, rₑ, uₑ, cell, local_face_index, face_cache, assembly_info)
        end
        assemble!(assembler, celldofs(cell), Jₑ, rₑ)

        # The assembly procedure might include local solves which have updated the local solution vector
        uncondense_ue!(u, uₑ, cell, assembly_info)
    end
end

"""
    mul!(out::AbstractVector, op::AssembledNonlinearOperator2, in::AbstractVector)
    mul!(out::AbstractVector, op::AssembledNonlinearOperator2, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out::AbstractVector, op::AssembledNonlinearOperator2, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::AssembledNonlinearOperator2, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)

Base.eltype(op::AssembledNonlinearOperator2) = eltype(op.J)
Base.size(op::AssembledNonlinearOperator2, axis) = size(op.J, axis)
