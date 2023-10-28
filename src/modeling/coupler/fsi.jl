"""
    LumpedFluidSolidCoupler

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
    ReggazoniSalvadorAfrica2022SurrogateVolume

Compute the chamber volume as a surface integral via the integral
  -∫ det(F) ((h ⊗ h)(x + d - b)) adj(F) N ∂Ωendo
  
Note that the integral basically measures the volume via displacement on a given axis.
"""
Base.@kwdef struct ReggazoniSalvadorAfrica2022SurrogateVolume{T}
    h::Vec{3,T} = Vec((0.0, 0.0, 1.0))
    b::Vec{3,T} = Vec((0.0, 0.0, 0.5))
end

# TODO split into 2 functions
function compute_chamber_volume(dh, u, method::ReggazoniSalvadorAfrica2022SurrogateVolume)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    volume = 0.0
    @unpack h, b = method
    drange = dof_range(dh,:displacement)
    for face ∈ FaceIterator(dh, getfaceset(grid, "Endocardium"))
        reinit!(fv, face)

        coords = getcoordinates(face)
        ddofs = @view celldofs(face)[drange]
        uₑ = @view u[ddofs]

        # ndofs_face = getnbasefunctions(fv)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            N = getnormal(fv, qp)

            ∇u = function_gradient(fv, qp, uₑ)
            F = one(∇u) + ∇u

            dq = function_value(fv, qp, uₑ)
            
            xq = spatial_coordinate(fv, qp, coords)

            volume -= det(F) * ((h ⊗ h) ⋅ (xq + dq - b)) ⋅ (transpose(inv(F)) ⋅  N) * dΓ
        end
    end
    return volume
end

function coupling_contribution!(Kc, dh, u, method::ReggazoniSalvadorAfrica2022SurrogateVolume)
    # TODO coupling assembler...
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :displacement)
    ip_geo = Ferrite.default_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_face = FaceQuadratureRule{ref_shape}(intorder)
    fv = FaceValues(qr_face, ip, ip_geo)

    @unpack h, b = method
    drange = dof_range(dh,:displacement)
    for face ∈ FaceIterator(dh, getfaceset(grid, "Endocardium"))
        reinit!(fv, face)

        coords = getcoordinates(face)
        ddofs = @view celldofs(face)[drange]
        uₑ = @view u[ddofs]

        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            N = getnormal(fv, qp)

            ∇u = function_gradient(fv, qp, uₑ)
            F = one(∇u) + ∇u

            dq = function_value(fv, qp, uₑ)

            xq = spatial_coordinate(fv, qp, coords)
            for j ∈ 1:getnbasefunctions(fv)
                Kc[1,j] -= det(F) * ((h ⊗ h) ⋅ (xq + dq - b)) ⋅ (transpose(inv(F)) ⋅  N) * dΓ
            end
        end
    end
    return volume
end
