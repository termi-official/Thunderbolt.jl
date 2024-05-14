"""
Enforce the constraints that
  chamber volume 3D (solid model) = chamber volume 0D (lumped circuit)
via Lagrange multiplied, where a surface pressure integral is introduced such that
  ∫  ∂Ωendo
Here `chamber_volume_method` is responsible to compute the 3D volume.
"""
struct LumpedFluidSolidCoupler{CVM} <: AbstractCoupler
    chamber_volume_method::CVM
end

is_bidrectional(::LumpedFluidSolidCoupler) = true

"""
    Debug helper for FSI. Just keeps the chamber volume constant.
"""
struct ConstantChamberVolume
    volume::Float64
end

function volume_integral(x, d, F, N, method::ConstantChamberVolume)
    method.volume
end

"""
Compute the chamber volume as a surface integral via the integral
  -∫ det(F) ((h ⊗ h)(x + d - b)) adj(F) N ∂Ωendo

as proposed by [RegSalAfrFedDedQar:2022:cem](@citet).

!!! note 
    This integral basically measures the volume via displacement on a given axis.
"""
Base.@kwdef struct RSAFDQ2022SurrogateVolume{T}
    h::Vec{3,T} = Vec((0.0, 0.0, 1.0))
    b::Vec{3,T} = Vec((0.0, 0.0, 0.5))
end

function volume_integral(x, d, F, N, method::RSAFDQ2022SurrogateVolume)
    @unpack h, b = method
    -det(F) * ((h ⊗ h) ⋅ (xq + dq - b)) ⋅ (transpose(inv(F)) ⋅  N)
end

"""
Chamber volume estimator as presented in [HirBasJagWilGee:2017:mcc](@cite).

Compute the chamber volume as a surface integral via the integral
 - ∫ (x + d) det(F) adj(F) N ∂Ωendo
where it is assumed that the chamber is convex, zero displacement in
apicobasal direction at the valvular plane occurs and the plane normal is aligned
with the z axis, where the origin is at z=0.
"""
struct Hirschvogel2017SurrogateVolume end

function volume_integral(x, d, F, N, method::Hirschvogel2017SurrogateVolume)
    -det(F) * (x + d) ⋅ inv(transpose(F)) ⋅ N
end

# TODO move these below into the opeartor interface once 
#      we figure out a good design on how to pass multiple
#      variables down to the inner assembly functions

function assemble_LFSI_coupling_contribution_row_inner!(C, R, u, p, face, dh, fv)
    reinit!(fv, face)

    drange = dof_range(dh, :displacement)

    coords = getcoordinates(face)
    ddofs = @view celldofs(face)[drange]
    uₑ = @view u[ddofs]

    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        d = function_value(fv, qp, uₑ)

        x = spatial_coordinate(fv, qp, coords)

        R[1] += volume_integral(x, u, F, N, method)
        # Via chain rule we obtain:
        #   δV(u,F(u)) = δu ⋅ dVdu + δF : dVdF
        ∂V∂u = Tensors.gradient(u -> volume_integral(x, u, F, N, method), d)
        ∂V∂F = Tensors.gradient(u -> volume_integral(x, d, u, N, method), F)
        for j ∈ 1:getnbasefunctions(fv)
            δuⱼ = shape_value(fv, qp, j)
            ∇δuj = shape_gradient(cv, qp, j)
            C[1, ddofs[j]] += (∂V∂u ⋅ δuⱼ + ∂V∂F ⊡ ∇δuj) * dΓ
        end
    end
end

"""
Chamber volume contribution for the 3D-0D constraint
    ∫ V³ᴰ(u) ∂Ω = V⁰ᴰ(c)
where u are the unkowns in the 3D problem and c the unkowns in the 0D problem.
"""
function assemble_LFSI_coupling_contribution_row!(C, R, dh, u, p, V⁰ᴰ, setname, method)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    for face ∈ FaceIterator(dh, getfaceset(grid, setname))
        assemble_LFSI_coupling_contribution_row_inner!(C, R, u, p, face, dh, fv)
    end

    R[1] -= V⁰ᴰ
end

function assemble_LFSI_coupling_contribution_col_inner!(C, R, u, p, face, dh, fv)
    reinit!(fv, face)
    drange = dof_range(dh, :displacement)

    ddofs = @view celldofs(face)[drange]
    uₑ = @view u[ddofs]

    for qp in QuadratureIterator(fv)
        ∂Ω₀ = getdetJdV(fv, qp)
        n₀ = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        invF = inv(F)
        cofF = transpose(invF)

        for j ∈ 1:getnbasefunctions(fv)
            δuⱼ = shape_value(fv, qp, j)
            R[ddofs[j]] += p * J * cofF ⋅ n₀ ⋅ δuⱼ * ∂Ω₀
            C[ddofs[j], 1] += J * cofF ⋅ n₀ ⋅ δuⱼ * ∂Ω₀
        end
    end
end

"""
Pressure contribution (i.e. variation w.r.t. p) for the term
    ∫ p n(u) δu ∂Ω
 [= ∫ p J(u) F(u)^-T n₀ δu ∂Ω₀]
where p is the unknown chamber pressure and u contains the unknown deformation field.
"""
function assemble_LFSI_coupling_contribution_col!(C, R, dh, u, p, setname, method)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    for face ∈ FaceIterator(dh, getfaceset(grid, setname))
        assemble_LFSI_coupling_contribution_col_inner!(C, R, u, p, face, dh, fv)
    end
end

function assemble_LFSI_volumetric_corrector_inner!(Kₑ::Matrix, residualₑ, uₑ, p, fv, method)
    reinit!(fv, face[1], face[2])

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        n₀ = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        # Add contribution to the residual from this test function
        # TODO fix the "nothing" here
        invF = inv(F)
        cofF = transpose(invF)
        J = det(F)
        neumann_term = p * J * cofF
        for i in 1:J
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ

            # ∂P∂Fδui =   ∂P∂F ⊡ (n₀ ⊗ δuᵢ) # Hoisted computation
            for j in 1:ndofs_face
                ∇δuⱼ = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                # Kₑ[i, j] += (n₀ ⊗ δuⱼ) ⊡ ∂P∂Fδui * dΓ
                #   δF^-1 = -F^-1 δF F^-1
                #   δJ = J tr(δF F^-1)
                # Product rule
                δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
                δJ = J * tr(∇δuⱼ ⋅ invF)
                δJcofF = δJ * cofF + J * δcofF
                Kₑ[i, j] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
            end
        end
    end
end

function assemble_LFSI_volumetric_corrector!(J, residual, dh, u, p, setname, method)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    drange = dof_range(dh,:displacement)

    assembler = start_assemble(J, residual, false)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)

    for face ∈ FaceIterator(dh, getfaceset(grid, setname))
        dofs = celldofs(face)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ .= @view u[dofs]
        assemble_LFSI_volumetric_corrector_inner!(Jₑ, rₑ, uₑ, p, fv, method)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end
end

function compute_chamber_volume(dh, u, setname, method)
    check_subdomains(dh)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    volume = 0.0
    drange = dof_range(dh,:displacement)
    for face ∈ FaceIterator(dh, getfaceset(grid, setname))
        reinit!(fv, face)

        coords = getcoordinates(face)
        ddofs = @view celldofs(face)[drange]
        uₑ = @view u[ddofs]

        for qp in QuadratureIterator(fv)
            dΓ = getdetJdV(fv, qp)
            N = getnormal(fv, qp)

            ∇u = function_gradient(fv, qp, uₑ)
            F = one(∇u) + ∇u

            d = function_value(fv, qp, uₑ)

            x = spatial_coordinate(fv, qp, coords)

            volume += volume_integral(x, d, F, N, method) * dΓ
        end
    end
    return volume
end

"""
Annotation for the split described by [RegSalAfrFedDedQar:2022:cem](@citet).
"""
struct RSAFDQSplit{MODEL <: CoupledModel}
    model::MODEL
end
