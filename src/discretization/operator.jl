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
    # TODO custom "square matrix tuple"
    operators::OPS # stored row by row as in [1 2; 3 4]
end

# TODO optimize
function getJ(op::BlockOperator, i::Block)
    @assert length(i.n) == 2
    mJs = reshape([getJ(opi) for opi ∈ op.operators], (isqrt(length(op.operators)), isqrt(length(op.operators))))
    return mJs[i.n[1], i.n[2]]
end

# TODO optimize
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
    function AssembledNonlinearOperator(J::MatrixType, element_cache::ElementCacheType, face_caches::FaceCacheType, dh::DHType) where {MatrixType, ElementCacheType, FaceCacheType, DHType <: AbstractDofHandler}
        check_subdomains(dh)
        return new{MatrixType, ElementCacheType, FaceCacheType, DHType}(J, element_cache, face_caches, dh)
    end
end

function setup_boundary_cache(boundary_models, qr::FaceQuadratureRule, ip, ip_geo)
    fv = FaceValues(qr, ip, ip_geo)
    return face_caches = ntuple(i->setup_face_cache(boundary_models[i], fv), length(boundary_models))
end

"""
    Utility constructor to get the nonlinear operator for a single field problem.
"""
function AssembledNonlinearOperator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FaceQuadratureRuleCollection)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.default_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    element_cache  = setup_element_cache(element_model, element_qr, ip, ip_geo)
    boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, ip, ip_geo)

    AssembledNonlinearOperator(
        create_sparsity_pattern(dh),
        element_cache,
        boundary_cache,
        dh,
    )
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

    assembler = start_assemble(J, residual)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)

    @inbounds for cell in CellIterator(dh)
        dofs = celldofs(cell)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ .= @view u[dofs]
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
        assemble!(assembler, dofs, Jₑ, rₑ)
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

struct AssembledBilinearOperator{MatrixType, CacheType, DHType <: AbstractDofHandler} <: AbstractBilinearOperator
    A::MatrixType
    element_cache::CacheType
    dh::DHType
    function AssembledBilinearOperator(A::MatrixType, element_cache::CacheType, dh::DHType) where {MatrixType, CacheType, DHType <: AbstractDofHandler}
        check_subdomains(dh)
        return new{MatrixType, CacheType, DHType}(A, element_cache, dh)
    end
end

function AssembledBilinearOperator(dh::AbstractDofHandler, field_name::Symbol, integrator, element_qrc::QuadratureRuleCollection)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the bilinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.default_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)

    element_cache = setup_element_cache(integrator, element_qr, ip, ip_geo)

    return AssembledBilinearOperator(
        create_sparsity_pattern(dh),
        element_cache,
        dh,
    )
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

update_linearization!(op::AbstractBilinearOperator, u, residual, time) = update_operator!(op, time)
update_linearization!(op::AbstractBilinearOperator, u, time) = update_operator!(op, time)

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

update_linearization!(::Thunderbolt.DiagonalOperator, ::Vector, ::Vector, t) = nothing

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

update_linearization!(::Thunderbolt.NullOperator, ::Vector, ::Vector, t) = nothing

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
Ferrite.add!(b::Vector, op::LinearNullOperator) = nothing
needs_update(op::LinearNullOperator, t) = false


struct LinearOperator{VectorType, CacheType, DHType <: AbstractDofHandler} <: AbstractLinearOperator
    b::VectorType
    element_cache::CacheType
    dh::DHType
    function LinearOperator(b::VectorType, element_cache::CacheType, dh::DHType) where {VectorType, CacheType, DHType <: AbstractDofHandler}
        check_subdomains(dh)
        return new{VectorType, CacheType, DHType}(b, element_cache, dh)
    end
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

Ferrite.add!(b::Vector, op::LinearOperator) = b .+= op.b
Base.eltype(op::LinearOperator) = eltype(op.b)
Base.size(op::LinearOperator) = sisze(op.b)

# TODO where to put these?
function create_linear_operator(dh, ::NoStimulationProtocol) 
    check_subdomains(dh)
    LinearNullOperator{Float64, ndofs(dh)}()
end
function create_linear_operator(dh, protocol::AnalyticalTransmembraneStimulationProtocol)
    check_subdomains(dh)
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
    _assemble_element!(bₑ, getcoordinates(cell), element_cache::AnalyticalCoefficientElementCache, time)
end
# We want this to be as fast as possible, so throw away everything unused
@inline function _assemble_element!(bₑ, coords::AbstractVector{<:Vec{dim,T}}, element_cache::AnalyticalCoefficientElementCache, time) where {dim,T}
    @unpack f, cv = element_cache
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    @inbounds for (qp, w) in pairs(Ferrite.getweights(cv.qr))
        # Compute dΩ
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, qp, coords)
        dΩ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping)) * w
        # Compute x
        x = spatial_coordinate(cv, qp, coords)
        # Evaluate f
        fx = f.f(x,time)
        # TODO replace with evaluate_coefficient
        # Evaluate all basis functions
        @inbounds for j ∈ 1:getnbasefunctions(cv)
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
