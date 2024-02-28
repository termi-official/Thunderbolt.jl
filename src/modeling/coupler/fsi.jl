"""
Enforce the constraints that
  chamber volume 3D (solid model) = chamber volume 0D (lumped circuit)
where a surface pressure integral is introduced such that
  ∫  ∂Ωendo
Here `chamber_volume_method` is responsible to compute the 3D volume.
"""
struct LumpedFluidSolidCoupler{CVM} <: InterfaceCoupler
    chamber_volume_method::CVM
end

"""
Compute the chamber volume as a surface integral via the integral
  -∫ det(F) ((h ⊗ h)(x + d - b)) adj(F) N ∂Ωendo

as proposed by [RegSalAfrFedDedQar:2022:cem](@citet).

!!! note 
    This integral basically measures the volume via displacement on a given axis.
"""
Base.@kwdef struct ReggazoniSalvadorAfrica2022SurrogateVolume{T}
    h::Vec{3,T} = Vec((0.0, 0.0, 1.0))
    b::Vec{3,T} = Vec((0.0, 0.0, 0.5))
end

function volume_integral(x, d, F, N, method::ReggazoniSalvadorAfrica2022SurrogateVolume)
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

function assemble_interface_coupling_contribution!(C, r, dh, u, setname, method::Hirschvogel2017SurrogateVolume)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    drange = dof_range(dh,:displacement)
    for face ∈ FaceIterator(dh, getfaceset(grid, setname))
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

            R[1,1] += volume_integral(x, d, F, N, method) * dΓ

            # δV(u,F(u)) = δ_u V + (δ_F V) ⋅ (δ_u F)
            ∂V∂u = Tensors.gradient(u -> volume_integral(x, u, F, N, method), d)
            ∂V∂F = Tensors.gradient(u -> volume_integral(x, d, u, N, method), F)
            for j ∈ 1:getnbasefunctions(fv)
                δuⱼ = shape_value(fv, qp, j)
                ∇δuj = shape_gradient(cv, qp, j)
                C[1, ddofs[j]] += (∂V∂u ⋅ δuⱼ + ∂V∂F ⊡ ∇δuj) * dΓ
            end
        end
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
struct ReggazoniSalvadorAfricaSplit{MODEL <: CoupledModel}
    model::MODEL
end
