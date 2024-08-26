using Ferrite, Tensors, UnPack

using SciMLBase#, DiffEqBase, OrdinaryDiffEq

# 0 = G(du,u,q,p,t)
# 0 = L(dq,u,q,p,t)
struct LocalGlobalFunction{GType, LTYpe} <: SciMLBase.AbstactNonlinearFunction
    G::GType
    L::LType
end

struct LocalGlobalProblem{F <: LocalGlobalFunction}
    f::F
end

# TODO citation of original paper and Hartmann's benchmark paper
struct MultilevelNewton
    max_iter::Int
end

struct MultilevelNewtonCache{VT, JT, LCTG, LCTL}
    max_iter::Int
    residual_global::VT
    residual_local::VT
    δu::VT
    J::JT
    lincache_global::LCTG # Global linear solver
    lincache_local::LCTL # Local linear solver
end

# function OrdinaryDiffEq.build_nlsolver(
#     alg, nlalg::MultilevelNewton,
#     u, uprev, p, t, dt,
#     f::F, rate_prototype, ::Type{uEltypeNoUnits},
#     ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
#     γ, c, α,
#     ::Val{true}) where {F, uEltypeNoUnits, uBottomEltypeNoUnits,
#     tTypeNoUnits}
# end

# struct QuasiStaticFunction{GType, LTYpe} <: OrdinaryDiffEq.AbstractDEFunction
#     G::GType
#     L::LType
# end

# struct QuasiStaticProblem{GType, LTYpe} <: OrdinaryDiffEq.AbstractDEFunction
#     G::GType
#     L::LType
# end

# function OrdinaryDiffEq.__solve(prob::QuasiStaticProblem)
# end

struct LinearViscoelasticity{T,TV,FT}
    K::T
    G::T
    viscosity::TV
    traction::FT
end

# Ferrite-specific
struct MaterialCache{M, DH, CV, FV}
    material::M
    dh::DH
    cv::CV
    fv::FV
end

function assemble_cell_Ju!(ke, u, q, cell, cellvalues, material::LinearViscoelasticity)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack Gᵥ   = material.viscosity
    @unpack K, G = material
    for q_point in 1:getnquadpoints(cellvalues)
        Eᵥ = q[q_offset+q_point]
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_symmetric_gradient(cellvalues, q_point, i)
            # Stress (Eq. 3)
            E = ∇Nᵢ
            I = one(E)
            T = K*tr(E)*I + 2*(G+Gᵥ)*dev(E) - 2*Gᵥ*dev(Eᵥ)
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇ˢʸᵐNⱼ ⊡ T) * dΩ
            end
        end
    end
    return nothing
end

function assemble_cell_Jq!(ke, u, q, cell, cellvalues, material::LinearViscoelasticity)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack Gᵥ   = material.viscosity
    @unpack K, G = material
    for q_point in 1:getnquadpoints(cellvalues)
        Eᵥ = q[q_offset+q_point]
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            C = gradient(ϵ -> -2*Gᵥ*dev(ϵ), Eᵥ);
            for j in 1:getnbasefunctions(cellvalues)
                # Gradient of the test function
                ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                ke[4*(q_point-1)+1, j] += (∇Nⱼ ⊡ C) * dΩ
            end
        end
    end
    return nothing
end

function assemble_cell!(ke, fe, u, q, cell, cellvalues, material::LinearViscoelasticity)
    assemble_cell_Ju!(ke, u, q, cell, cellvalues, material)
    return nothing
end

# Standard assembly loop for global problem
function stress!(du, u, q, cache::MaterialCache, t)
    K = allocate_matrix(cache.dh)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        # Compute element contribution
        assemble_cell!(ke, fe, cell, cache.cv, material)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    for face in FacetIterator(dh, "right")
        # Update the facetvalues to the correct facet number
        reinit!(cache.fv, face)
        # Reset the temporary array for the next facet
        fill!(fe, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(face)
        for qp in 1:getnquadpoints(cache.fv)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(cache.fv, qp, cell_coordinates)
            tₚ = cache.material.traction(x)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(cache.fv, qp)
            for i in 1:getnbasefunctions(cache.fv)
                Nᵢ = shape_value(cache.fv, qp, i)
                fe_ext[i] -= tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f_ext, celldofs(face), fe_ext)
    end
    du .= K*u .- f
end

function stress_Ju!(K, u, q, cache::MaterialCache, t)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        # Compute element contribution
        assemble_cell_Ju!(ke, u, q, cell, cache.cv, cache.material)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
end

# function stress_Jq!(K, u, q, material::MaterialCache, t)
#     # Allocate the element stiffness matrix
#     n_basefuncs = getnbasefunctions(cache.cv)
#     ke = zeros(n_basefuncs, n_basefuncs)
#     # Create an assembler
#     assembler = start_assemble(K)
#     # Loop over all cells
#     for cell in CellIterator(dh)
#         # Update the shape function gradients based on the cell coordinates
#         reinit!(cache.cv, cell)
#         # Reset the element stiffness matrix
#         fill!(ke, 0.0)
#         # Compute element contribution
#         assemble_cell_Jq!(ke, cell, cache.cv, material)
#         # Assemble ke into K
#         assemble!(assembler, celldofs(cell), ke)
#     end
# end

struct OverstressedViscosity{T}
    Gᵥ::T
    η₀::T
    s₀::T
end

struct FerriteLocalChunkInfo{CV, QP}
    cv::CV
    qpᵢ::QP
end

function viscosity_evolution!(dq, u, q, p::OverstressedViscosity, t, local_chunk_info)
    @unpack Gᵥ, η₀, s₀ = p
    @unpack cv, qpᵢ = local_chunk_info

    E   = symmetric_function_gradient(u, cv)
    Eᵥ  = q
    Tₒᵥ = 2Gᵥ * dev(E-Eᵥ)
    η   = η₀ * exp(-s₀*norm(Tₒᵥ))

    dq .= Tₒᵥ/η
end

function viscosity_evolution_Jq(u, q, p::OverstressedViscosity, t, local_chunk_info)
    @unpack Gᵥ, η₀, s₀ = p
    @unpack cv = local_chunk_info

    E   = symmetric_function_gradient(u, cv)
    Tₒᵥ(Eᵥ) = 2Gᵥ * dev(E-Eᵥ)
    η(Eᵥ)   = η₀ * exp(-s₀*norm(Tₒᵥ))

    return gradient(Eᵥ -> Tₒᵥ(Eᵥ)/η(Eᵥ), q); 
end

function viscosity_evolution_Ju!(Ju, u, q, p::OverstressedViscosity, t, local_chunk_info)
    @unpack Gᵥ, η₀, s₀ = p
    @unpack cv, qpᵢ = local_chunk_info

    E   = symmetric_function_gradient(u, cv)
    Eᵥ  = q
    Tₒᵥ = 2Gᵥ * dev(E-Eᵥ)
    η   = η₀ * exp(-s₀*norm(Tₒᵥ))

    dq .= Tₒᵥ/η
end

# function solve_local()
#     # for each local chunk # Element loop
#     #     for each element in chunk # Quadrature loop
#     #         solve stage ,,L(uₗ, qₗ) = 0'' with fixed uₗ for qₗ
#     #         solve dQ/dU = [∂L/∂Q]^-1 ∂L/∂
#     #         update ∂G/∂U with dQ/dU
#     #     end
#     # end
# end

function traction(x)
    return Vec(0.0, 1.0 * x[2])
end

# Hartmann 2005 Table 5
material = LinearViscoelasticity(
    1667.00,
       8.75,
    OverstressedViscosity(
        150.0,
        13.0
        1.1,
    ),
    traction,
)

u₀ = zeros(ndofs(dh))
q₀ = zeros(getnquadpoints(cv)*getncells(dh.grid))

dt = 1.0
T  = 10.0


function generate_linear_elasticity_problem()
    grid = generate_grid(Quadrilateral, (1, 1));

    dim = 2
    order = 1 # linear interpolation
    ip = Lagrange{RefQuadrilateral, order}()^dim; # vector valued interpolation

    qr = QuadratureRule{RefQuadrilateral}(2)
    qr_face = FacetQuadratureRule{RefQuadrilateral}(2);

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0, [1,2]))
    close!(ch);

    return dh, ch, cellvalues, facetvalues
end

# init
dh,ch,cv,fv = generate_linear_elasticity_problem()
J = allocate_matrix(dh)
global_residual = zeros(ndofs(dh))
u = copy(u₀)
q = copy(q₀)
cache = MaterialCache(
    material,
    dh,
    cv,
    fv,
)

# solve
for t ∈ 0.0:dt:T
    for iter in 1:10
        #Compute residual
        stress!(global_residual, u, q, cache::MaterialCache, t+dt)

        #Check convergence
        rnorm = norm(global_residual)
        @info "$t: Iteration $iter $rnorm"
        rnorm < 1e-4 && break

        #Setup Jacobian
        # 1. Global part
        stress_Ju!(J, u, q, cache, t+dt)
        # 2. Local corrections
        # for each local chunk # Element loop
        for cell in CellIterator(dh)
            # prepare iteration
            reinit!(cache.cv, cell)
            ue = @views u[celldofs(cell)] # TODO copy for better cache utilization
            q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
            local_chunk_info = FerriteLocalChunkInfo(cache.cv, qp)
            # for each item in local chunk # Quadrature loop
            for qp in 1:getnquadpoints(cache.cv)
                # solve local problem
                for inner_iter in 1:10
                    q[q_offset+qp] .+= ...?
                end
                # update Jacobian
                invlocal_Jq = inv(viscosity_evolution_Jq(ue, q[q_offset+qp], cache.material.viscosity, t+dt, local_chunk_info))
                for i in 1:getnbasefunctions(cv)
                    for j in 1:getnbasefunctions(cv)
                        invlocal_Jq ⊡ local_residual
                    end
                end
            end
        end
        # 3. Boundary conditions
        apply!(J, global_residual, ch)
        # 4. Solve linear system
        Δu = J \ global_residual
        # 5. Update solution
        u .+= Δu
        iter == 10 && error("max iter")
    end
end
