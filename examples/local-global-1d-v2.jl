using Thunderbolt, UnPack

import Thunderbolt.WriteVTK: paraview_collection 

struct LinearViscosity1D{T}
    E₁::T
    η₁::T
end

struct LinearViscoelasticity1D{T,TV,FT}
    E₀::T
    viscosity::TV
    traction::FT
end

# Ferrite-specific
struct ElementCache{M, DH, CV, FV}
    material::M
    dh::DH
    cv::CV
    fv::FV
end

abstract type AbstractStageCache end

struct BackwardEulerCache{T, QT} <: AbstractStageCache
    t::T
    dt::T
    qprev::QT
    tol::T
end

struct LocalInfoCache
    qp::Int
end

function solve_local_problem(u, q, model, solver_cache::BackwardEulerCache, info::LocalInfoCache)
    @unpack t, dt, tol = solver_cache
    # TODO how to handle this correctly? Should be somehow part of the local solver cache?
    @unpack qp = info
    qprev = @view solver_cache.qprev[qp]
    # solve local problem
    for inner_iter in 1:max_iter
        # setup system for local backward Euler solve L(qₙ₊₁, uₙ₊₁) = (qₙ₊₁ - qₙ) - Δt*rhs(qₙ₊₁, uₙ₊₁) = 0 w.r.t. qₙ₊₁ with uₙ₊₁ frozen
        J        = jacobian_q(u, q, model, t+dt)
        dq       = f(u, q, cache.material.viscosity, t+dt)
        residual = (q .- qprev) .- dt .* dq # = L(u,q)
        # solve linear sytem and update local solution
        ∂L∂q     = one(J) - (dt .* J)
        Δq       = ∂L∂q \ residual
        q       -= Δq
        # check convergence
        resnorm  = norm(residual)
        (resnorm ≤ tol || norm(Δq) ≤ tol) && break
        # inner_iter == max_iter && error("Newton diverged for cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
        inner_iter == max_iter && error("qp=$qp at t=$t resnorm=$resnorm")
    end
    return q
end

function solve_corrector_problem(E, Eᵥ, model, solver_cache::BackwardEulerCache)
    @unpack t, dt = solver_cache
    J    = jacobian_q(E, Eᵥ, model, t+dt)
    ∂L∂Q = (one(J) - (dt .* J))
    ∂L∂U = -dt * jacobian_u(E, Eᵥ, model, t+dt)
    dQdU = zeros(length(∂L∂U))
    return dQdU = ∂L∂Q \ -∂L∂U
end

function assemble_cell!(ke, fe, u, q, cell, cellvalues, material::LinearViscoelasticity1D, solver_cache::AbstractStageCache)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack E₁   = material.viscosity
    @unpack E₀   = material
    ke2 = zeros(2,2)
    for q_point in 1:getnquadpoints(cellvalues)
        # 1. Solve for Qₙ₊₁
        E  = function_gradient(cellvalues, q_point, u)[1]
        Eᵥ = q[q_offset+q_point] # Current guess
        Eᵥ = solve_local_problem(E, Eᵥ, material.viscosity, solver_cache, LocalInfoCache(q_offset+q_point))
        q[q_offset+q_point] = Eᵥ # Update guess
        # 2. Contribution to corrector part ,,dQdU`` by solving the linear system
        dQdU = solve_corrector_problem(E, Eᵥ, material.viscosity, solver_cache)
        # 3. ∂G∂Q
        ∂G∂Q = -E₁
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)[1]
            # T = E₀*∇Nᵢ + E₁*(∇Nᵢ-Eᵥ)
            T = (E₀ + E₁)*∇Nᵢ
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_gradient(cellvalues, q_point, j)[1]
                ke[i, j] += (∇ˢʸᵐNⱼ * T) * dΩ
                # Corrector part
                ke[i, j] += ∇ˢʸᵐNⱼ * ∂G∂Q * dQdU * ∇Nᵢ * dΩ
                ke2[i, j] += ∇ˢʸᵐNⱼ * ∂G∂Q * dQdU * ∇Nᵢ * dΩ
            end
        end
    end
    return nothing
end

# Standard assembly loop for global problem
function stress!(J, residual, u, q, cache::ElementCache, solver_cache::AbstractStageCache)
    f = zeros(ndofs(cache.dh))
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Create an assembler
    assembler = start_assemble(J, residual)
    # Loop over all cells
    for cell in CellIterator(cache.dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        # Compute element contribution
        assemble_cell!(ke, fe, u, q, cell, cache.cv, cache.material, solver_cache)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    for face in FacetIterator(cache.dh, getfacetset(cache.dh.grid,"right"))
        # Update the facetvalues to the correct facet number
        reinit!(cache.fv, face)
        # Reset the temporary array for the next facet
        fill!(fe, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(face)
        for qp in 1:getnquadpoints(cache.fv)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(cache.fv, qp, cell_coordinates)
            tₚ = cache.material.traction(x, solver_cache.t+solver_cache.dt)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(cache.fv, qp)
            for i in 1:getnbasefunctions(cache.fv)
                Nᵢ = shape_value(cache.fv, qp, i)
                fe[i] += tₚ * Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f, celldofs(face), fe)
    end
    residual .= J*u .- f
end

function viscosity_evolution(E, Eᵥ, p::LinearViscosity1D, t)
    @unpack E₁, η₁ = p
    return E₁/η₁*(E-Eᵥ)
end

function viscosity_evolution_Jq(E, Eᵥ, p::LinearViscosity1D, t)
    @unpack E₁, η₁ = p
    return -E₁/η₁
end

function viscosity_evolution_Ju(E, Eᵥ, p::LinearViscosity1D, t)
    @unpack E₁, η₁ = p
    return E₁/η₁
end

f          = viscosity_evolution
jacobian_u = viscosity_evolution_Ju
jacobian_q = viscosity_evolution_Jq

function traction(x, t)
    return 0.1
end

# Hartmann 2005 Table 5
material = LinearViscoelasticity1D(
    1.0, #70
    LinearViscosity1D(
        1.0,#20
        1.0,
    ),
    traction,
)

function generate_linear_elasticity_problem()
    grid = generate_grid(Line, (1,));

    order = 1 # linear interpolation
    ip = Lagrange{RefLine, order}() # vector valued interpolation

    qr = QuadratureRule{RefLine}(2)
    qr_face = FacetQuadratureRule{RefLine}(1);

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0))
    # add!(ch, Dirichlet(:u, getfacetset(grid, "right"),   (x, t) -> 0.1))
    close!(ch);

    return dh, ch, cellvalues, facetvalues
end

dh,ch,cv,fv = generate_linear_elasticity_problem()
cache = ElementCache(
    material,
    dh,
    cv,
    fv,
)
u₀ = zeros(ndofs(dh))
q₀ = zeros(getnquadpoints(cv)*getncells(dh.grid))

dt = 1.0
T  = 2.0

# init
J = allocate_matrix(dh)
global_residual = zeros(ndofs(dh))
u = copy(u₀)
q = copy(q₀)
qprev = copy(q₀)

max_iter   = 100 # Should be 1
local_tol⁰ = 1e-16
global_tol = 1e-8

# solve
Δu = zeros(ndofs(dh))
# Translation Ferrite->SciML
global_fJu        = stress!

pvd = paraview_collection("linear-viscoelasticity")
for t ∈ 0.0:dt:T
    # For each stage ... this is a backward Euler for now, so just 1 stage :)
    # Outer newton loop
    update!(ch, t)
    for outer_iter in 1:max_iter
        # Apply constraints to solution
        apply!(u, ch)

        #Setup Jacobian
        # 1. Local solves and Jacobian corrections (normally these are done IN global_jacobian_u for efficiency reasons)
        # Done in 2.
        # 2. Residual and Jacobian of global function G(qₙ₊₁, uₙ₊₁) w.r.t. to global vector uₙ₊₁ with qₙ₊₁ frozen
        local_tol = outer_iter == 1 ? local_tol⁰ : min(local_tol⁰, norm(Δu)^2)
        local_solver = BackwardEulerCache(t, dt, qprev, local_tol)
        global_fJu(J, global_residual, u, q, cache, local_solver)
        @show J, global_residual
        # 3. Apply boundary conditions
        apply_zero!(J, global_residual, ch)
        # 4. Solve linear system
        Δu .= J \ global_residual
        # 5. Update solution
        u .-= Δu
        #Check convergence
        rnorm = norm(global_residual)
        inorm = norm(Δu)
        @info "$t: Iteration=$outer_iter rnorm=$rnorm inorm=$inorm"
        if rnorm ≤ global_tol
            break
        end
        if norm(Δu) ≤ global_tol
            break
        end
        outer_iter == max_iter && error("max iter")
    end
    @show q
    qprev .= q

    VTKGridFile("linear-viscoelasticity-$t.vtu", dh) do vtk
        write_solution(vtk, dh, u)
        # for (i, key) in enumerate(("11", "22", "12"))
        #     write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
        # end
        # write_projection(vtk, proj, stress_field, "stress field")
        # Ferrite.write_cellset(vtk, grid)
        pvd[t] = vtk
    end
end
# close(pvd)
