using Ferrite, Tensors, ProgressMeter, BlockArrays, UnPack, DelimitedFiles, Plots

abstract type HyperelasticMaterial end

Base.@kwdef struct CoupledQuasiCompressNeoHookeanMaterial <: HyperelasticMaterial
    E = 1.0
    ν = 0.3
end

function Ψ(F, mp::CoupledQuasiCompressNeoHookeanMaterial)
	@unpack E, ν = mp
    J = det(F)
	C = tdot(F)
    I₁ = tr(C)

    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))

	return μ / 2 * (I₁ - 3) - μ * log(J) + λ / 2 * log(J)^2
end

Base.@kwdef struct LinearSpringMaterial <: HyperelasticMaterial
    η = 1.0
    dir::Vec{3}
end

function Ψ(F, mp::LinearSpringMaterial)
	@unpack η, dir = mp
	C = tdot(F)
    I₄ = dir ⋅ C ⋅ dir

	return η/2 * (I₄ - 1)^2
end

abstract type ContractionUnitModel end

struct Pelce1995ContractionUnitModel <: ContractionUnitModel end

Base.@kwdef struct PPMK2017ContractionUnitModel <: ContractionUnitModel
    v₀ = 3.0 # 1/s
    β  = 4.0
    F₀ = Ca -> Ca
end

function ψstar(Q̇, Ca, mp::PPMK2017ContractionUnitModel)
    @unpack v₀, β, F₀ = mp
    return F₀(Ca)*v₀/β * exp(β*Q̇/v₀)
end

# Evaluation of min function apprearing in (47) without substraction
function Ψmin_reduced(Fₖ₊₁, Fᵃₖ, Qₖ₊₁, Qₖ, Δt, Ca, mp, f₀, s₀, n₀)
    # Apply flow rule (44)
    ΔQ = Qₖ₊₁ - Qₖ
    Fᵃₖ₊₁ = (exp(ΔQ)*f₀⊗f₀ + s₀⊗s₀ + n₀⊗n₀)⋅Fᵃₖ
    Fᵉₖ₊₁ = Fₖ₊₁⋅inv(Fᵃₖ₊₁)

    return Ψ(Fₖ₊₁, mp.P) + Ψ(Fᵉₖ₊₁, mp.A) + Δt * ψstar(ΔQ/Δt, Ca, mp.C)
end

# Evaluation of min function apprearing in (47)
function Ψmin(Fₖ₊₁, Fᵃₖ, Fₖ, Qₖ₊₁, Qₖ, Δt, Ca, mp, f₀, s₀, n₀)
    # Apply flow rule (44)
    ΔQ = Qₖ₊₁ - Qₖ
    Fᵃₖ₊₁ = (exp(ΔQ)*f₀⊗f₀ + s₀⊗s₀ + n₀⊗n₀)*Fᵃₖ
    Fᵉₖ₊₁ = Fₖ₊₁*inv(Fᵃₖ₊₁)

    return Ψ(Fₖ₊₁, mp.P) + Ψ(Fᵉₖ₊₁, mp.A) - Ψ(Fₖ, mp.P) - Ψ(Fᵉₖ, mp.A) + Δt * ψstar(ΔQ/Δt, Ca, mp.C)
end

function predictor(Fₖ₊₁, Fᵃₖ, Qₖ, mp, Δt, Ca, f₀, s₀, n₀)
    Qₖ₊₁ = copy(Qₖ) # initial guess
    ΔQ = 0.0 # Newton increment
    # Newton loop for minimization in (47)
    for i = 1:10
        # println("Internal iteration $i with energy ", Ψmin_reduced(Fₖ₊₁, Fᵃₖ, Qₖ₊₁, Qₖ, Δt, Ca, mp, f₀, s₀, n₀))
        jacobian, residual = Tensors.hessian(Qₖ₊₁_ -> Ψmin_reduced(Fₖ₊₁, Fᵃₖ, Qₖ₊₁_, Qₖ, Δt, Ca, mp, f₀, s₀, n₀), Qₖ₊₁, :all)
        ΔQ = jacobian \ -residual
        Qₖ₊₁ += ΔQ

        if norm(ΔQ) < 1e-6
            return Qₖ₊₁
        end
    end
    println("Predictor failed to converge with energy ", Ψmin_reduced(Fₖ₊₁, Fᵃₖ, Qₖ₊₁, Qₖ, Δt, Ca, mp, f₀, s₀, n₀))
    return Qₖ₊₁
end



"""
Generalized Hill modela s described in Göktepe, Menzel, Kuhl (2014).

  +--Ψᵖ--+
  |      |
--+      +--
  |      |
  +-Ψᵃ-C-+

"""
struct GeneralizedHillMaterial{PM,AM,CUM} #where {PM <: HyperelasticMaterial,AM <: HyperelasticMaterial,CUM <: ContractionUnitModel}
    P::PM # passive material
    A::AM # active material
    C::CUM # contraction unit's model
end

#
function incremental_driver(Fₙ₊₁, Fₙ, Qₙ, mp::GeneralizedHillMaterial{PM,AM,PPMK2017ContractionUnitModel}) where {PM <: HyperelasticMaterial, AM <: HyperelasticMaterial}
    Qₙ₊₁ = internal_driver(Fₙ₊₁, Qₙ, mp.C)

    # Compute all derivatives in one function call
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    P = ∂Ψ∂F
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

function assemble_element!(ke, ge, cell, cv, fv, mp::HyperelasticMaterial, ue, ΓN)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    tn = 0.0 # Traction (to be scaled with surface normal)
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
        # Compute stress and tangent
        P, ∂P∂F = constitutive_driver(F, Fᵃ, mp)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += ( ∇δui ⊡ P - δui ⋅ b ) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end

    # Surface integral for the traction
    for face in 1:nfaces(cell)
        if (cellid(cell), face) in ΓN
            reinit!(fv, cell, face)
            for q_point in 1:getnquadpoints(fv)
                t = tn * getnormal(fv, q_point)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
end


function assemble_element!(ke, ge, cell, cv, fv, mp::GeneralizedHillMaterial, ue, ΓN)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    tn = 0.0 # Traction (to be scaled with surface normal)
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
        # Compute stress and tangent
        P, ∂P∂F = constitutive_driver(F, Fᵃ, mp)

        # Qₙ₊₁ = predictor(Fₖ₊₁, Fᵃₖ, Qₖ, mp, Δt, f₀, s₀, n₀)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += ( ∇δui ⊡ P - δui ⋅ b ) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end

    # Surface integral for the traction
    for face in 1:nfaces(cell)
        if (cellid(cell), face) in ΓN
            reinit!(fv, cell, face)
            for q_point in 1:getnquadpoints(fv)
                t = tn * getnormal(fv, q_point)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
end

function assemble_global!(K, f, dh, cv, fv, mp, u, ΓN)
    n_total = getnbasefunctions(cellvalues)
    Ke = zeros(ntotal, ntotal)
    ge = zeros(ntotal)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
        assemble!(assembler, global_dofs, ge, ke)
    end
end;

function solve(Δt = 0.01, T = 1.0)
    dim = 3
    order = 1

    # Generate a grid
    N = 10
    L = 1.0
    left = zero(Vec{dim})
    right = L * ones(Vec{dim})
    grid = generate_grid(Hexahedron, ntuple(_->N, dim), left, right)

    # Material parameters
    E = 10.0
    ν = 0.3
    mp = CoupledQuasiCompressNeoHookeanMaterial(E, ν)

    # Finite element base
    ip = Lagrange{dim, RefCube, order}()
    qr = QuadratureRule{dim, RefCube}(2*order-1)
    qr_face = QuadratureRule{dim-1, RefCube}(2*order-1)
    cv = CellVectorValues(qr, ip)
    fv = FaceVectorValues(qr_face, ip)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    function rotation(X, t, θ = deg2rad(60.0), T = T)
        x, y, z = X
        return t * Vec{dim}(
            (0.0,
            t/T*(L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ)),
            t/T*(L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ))
            ))
    end

    dbcs = ConstraintHandler(dh)
    # Add a homogenous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> rotation(x, t), [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)

    # Neumann part of the boundary
    ΓN = union(
        getfaceset(grid, "top"),
        getfaceset(grid, "bottom"),
        getfaceset(grid, "front"),
        getfaceset(grid, "back"),
    )

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)

    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
    for t ∈ 0.0:Δt:T
        Ferrite.update!(dbcs, t)
        apply!(un, dbcs)

        # Create sparse matrix and residual vector
        K = create_sparsity_pattern(dh)
        g = zeros(_ndofs)

        # Perform Newton iterations
        newton_itr = -1
        NEWTON_TOL = 1e-8
        NEWTON_MAXITER = 30

        # Store new last solution
        un .= u

        while true; newton_itr += 1
            u .= un .+ Δu # Current guess

            assemble_global!(J, g, dh, cv, fv, mp, u, ΓN)
            normg = norm(g[Ferrite.free_dofs(dbcs)])
            apply_zero!(J, g, dbcs)

            ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])

            if normg < NEWTON_TOL
                break
            elseif newton_itr > NEWTON_MAXITER
                error("Reached maximum Newton iterations, aborting")
            end

            ΔΔu = J \ g;

            apply_zero!(ΔΔu, dbcs)
            Δu .-= ΔΔu
        end

        # Save the solution
        vtk_grid("ppmk2017-t=$t", dh) do vtkfile
            vtk_point_data(vtkfile, dh, u)
        end
    end
end

# solve();

function solve_qp()
    f₀ = Vec{3}((1.0, 0.0, 0.0))
    s₀ = Vec{3}((0.0, 1.0, 0.0))
    n₀ = Vec{3}((0.0, 0.0, 1.0))

    # Material parameters
    E = 1.0
    ν = 0.3
    η = 10.0
    mp = GeneralizedHillMaterial(
        CoupledQuasiCompressNeoHookeanMaterial(E, ν),
        LinearSpringMaterial(η, f₀),
        PPMK2017ContractionUnitModel()
    )

    Qₖ = 0.0
    Qₖ₊₁ = 0.0
    ΔQ = 0.0

    Fₖ = one(Tensor{2,3})
    Fₖ₊₁ = one(Tensor{2,3})
    Fᵃₖ₊₁ = one(Tensor{2,3})

    # u = zeros(3)
    # un = zeros(3)
    # Δu = zeros(3)

    newton_itr = -1
    NEWTON_TOL = 1e-8
    NEWTON_MAXITER = 30

    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
    t_prev = 0.0
    
    t_results = Vector{Real}(undef,0)
    F_results = Vector{Tensor{2,3}}(undef,0)
    Q_results = Vector{Real}(undef,0)
    for (i,(t,Ca)) ∈ enumerate(eachrow(readdlm("data/tests/mahajan-shiferaw-calcium.dat", ' ', Float64, '\n')))
        if i == 1
            continue;
        end
        Δt = t - t_prev
        # Perform Newton iterations
        Qₖ = Qₖ₊₁
        #uₖ .= uₖ₊₁
        Fₖ = Fₖ₊₁
        Fᵃₖ = Fᵃₖ₊₁

        while true; newton_itr += 1
            #u .= un .+ Δu # Current guess

            # Update internal variable
            Qₖ₊₁ = predictor(Fₖ₊₁, Fᵃₖ, Qₖ, mp, Δt, Ca, f₀, s₀, n₀)

            #TODO Rewrite with arrays to handle higher dimensional Q
            ∂W_F = (Qₖ₊₁_) -> Tensors.gradient(Fₖ₊₁_ -> Ψmin_reduced(Fₖ₊₁_, Fᵃₖ, Qₖ₊₁_, Qₖ, Δt, Ca, mp, f₀, s₀, n₀), Fₖ₊₁)
            ∂W_Q = (Fₖ₊₁_) -> Tensors.gradient(Qₖ₊₁_ -> Ψmin_reduced(Fₖ₊₁_, Fᵃₖ, Qₖ₊₁_, Qₖ, Δt, Ca, mp, f₀, s₀, n₀), Qₖ₊₁)

            # Eval second derivatives
            ∂²W∂F² = Tensors.hessian(Fₖ₊₁_ -> Ψmin_reduced(Fₖ₊₁_, Fᵃₖ, Qₖ₊₁, Qₖ, Δt, Ca, mp, f₀, s₀, n₀), Fₖ₊₁)
            # ∂²W∂Q² = Tensors.gradient(Qₖ₊₁_ -> ∂W_Q(Fₖ₊₁, Qₖ₊₁_), Qₖ₊₁)
            ∂²W∂Q² = Tensors.hessian(Qₖ₊₁_ -> Ψmin_reduced(Fₖ₊₁, Fᵃₖ, Qₖ₊₁_, Qₖ, Δt, Ca, mp, f₀, s₀, n₀), Qₖ₊₁)
            ∂²W∂F∂Q = Tensors.gradient(Qₖ₊₁_ -> ∂W_F(Qₖ₊₁_), Qₖ₊₁)
            ∂²W∂Q∂F = Tensors.gradient(Fₖ₊₁_ -> ∂W_Q(Fₖ₊₁_), Fₖ₊₁)

        
            #println("det(∂²W∂Q²) = ", det(∂²W∂Q²))# check for invertibility

            #ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])

            # if normg < NEWTON_TOL
            #     break
            # elseif newton_itr > NEWTON_MAXITER
            #     error("Reached maximum Newton iterations, aborting")
            # end

            # ΔΔu = J \ g;

            # Δu .-= ΔΔu

            ΔQ = Qₖ₊₁ - Qₖ
            Fᵃₖ₊₁ = (exp(ΔQ)*f₀⊗f₀ + s₀⊗s₀ + n₀⊗n₀)⋅Fᵃₖ

            break;
        end

        push!(t_results, t)
        push!(Q_results,  Qₖ₊₁)
        push!(F_results, Fᵃₖ₊₁)

        t_prev = t
    end
    l = @layout [a ; b]
    p1 = plot(t_results, [F[1,1] for F ∈ F_results])
    p2 = plot(t_results, Q_results)
    plot(p1, p2, layout = l)
end

solve_qp();