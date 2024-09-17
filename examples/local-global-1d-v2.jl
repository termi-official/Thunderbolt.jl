using Thunderbolt, UnPack

import Thunderbolt.WriteVTK: paraview_collection 

# Local out-of-place functions of pure ODE form dq = f(u,q,p,t)
abstract type LocalODEModel end

# -----------vvv HERE WE DEFINE THE LOCAL MODEL vvv----------------
struct LinearViscosity1D{T} <: LocalODEModel
    E₁::T
    η₁::T
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

# Transformation from vector to tensor form
function q_to_local_form(q, model::LinearViscosity1D)
    return q[1]
end

# Transformation from tensor form to vector
function store_local_form_q_to!(qstorage, qform, model::LinearViscosity1D)
    return qstorage[1] = qform
end
# -----------^^^ HERE WE DEFINE THE LOCAL MODEL ^^^----------------

# -----------vvv HERE WE DEFINE THE GLOBAL MODEL vvv----------------
struct LinearViscoelasticity1D{T,TV,FT}
    E₀::T
    viscosity::TV
    traction::FT
end
# -----------vvv HERE WE DEFINE THE GLOBAL MODEL vvv----------------

# Ferrite-specific variables
struct StructuralElementCache{M, DH, CV, FV}
    material::M
    dh::DH
    cv::CV
    fv::FV
end

struct QuasiStaticSolutionState{UT, QT}
    u::UT
    q::QT
end

# struct DynamicSolutionState{UT, VT, QT}
#     u::UT
#     v::VT
#     q::QT
# end

abstract type AbstractStageCache end

struct BackwardEulerStage{T, ST} <: AbstractStageCache
    t::T   # Integrator
    dt::T  # Integrator
    solprev::ST
    # Multilevel solver info
    tol::T
    max_iter::Int
end

struct LocalBackwardEulerCache{T, ST, ST2}
    t::T   # Integrator
    dt::T  # Integrator
    sol::ST
    solprev::ST2
    # NLSolver
    tol::T
    max_iter::Int
end

struct QuadraturePointStageInfo{iType, CV}
    qpi::iType
    cellvalues::CV
    # TODO QuadratureHandler
end

# MUST NO ALLOCATE (to support GPU)
function build_local_cache(sol::QuasiStaticSolutionState, cell::CellCache, info::QuadraturePointStageInfo, stage::BackwardEulerStage)
    @unpack cellvalues = info
    udofs = celldofs(cell)
    uprev = @view stage.solprev.u[udofs]
    Eprev = function_gradient(cellvalues, info.qpi, uprev)[1]

    # qdofs = @view quadraturedofs(...) # TODO implement and use this instead of the manual extration below
    sol_size_local = length(sol.q)
    nqp            = getnquadpoints(cellvalues)*getncells(cell.grid)
    qprev_t3 = reshape(stage.solprev.q, (sol_size_local, nqp))
    qoffset  = getnquadpoints(cellvalues)*(cellid(cell)-1)
    qprev    = @view qprev_t3[:, qoffset + info.qpi]

    # Wrap into correct solution type
    solprev = QuasiStaticSolutionState(
        Eprev,
        qprev,
    )

    # Construct local cache object to control the dispatches
    return LocalBackwardEulerCache(
        stage.t,
        stage.dt,
        sol,
        solprev,
        stage.tol,
        stage.max_iter,
    )
end

function local_problem_residual(cache::LocalBackwardEulerCache, model::LocalODEModel)
    @unpack t, dt = cache
    u     = cache.sol.u
    uprev = cache.solprev.u
    q     = q_to_local_form(cache.sol.q, model)
    qprev = q_to_local_form(cache.solprev.q, model)
    dq       = local_rhs(u, q, model, t+dt)
    residual = (q .- qprev) .- dt .* dq # = L(u,q)
    return residual # FIXME
end

function solve_local_problem(cache::LocalBackwardEulerCache, model::LocalODEModel)
    @unpack t, dt, tol = cache
    u     = cache.sol.u
    uprev = cache.solprev.u
    q     = q_to_local_form(cache.sol.q, model)
    qprev = q_to_local_form(cache.solprev.q, model)
    # du = (u - uprev)/dt
    # solve local problem with Newton-Raphson
    for inner_iter in 1:max_iter
        # setup system for local backward Euler solve L(qₙ₊₁, uₙ₊₁) = (qₙ₊₁ - qₙ) - Δt*rhs(qₙ₊₁, uₙ₊₁) = 0 w.r.t. qₙ₊₁ with uₙ₊₁ frozen
        residual = local_problem_residual(cache, model) # = L(u,q)
        # solve linear sytem and update local solution
        J        = local_∂q(u, q, model, t+dt)
        ∂L∂q     = one(J) - (dt .* J)
        Δq       = ∂L∂q \ residual
        q       -= Δq
        # Update cache
        store_local_form_q_to!(cache.sol.q, q, model)
        # check convergence
        resnorm  = norm(residual)
        (resnorm ≤ tol || norm(Δq) ≤ tol) && break
        # inner_iter == max_iter && error("Newton diverged for cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
        inner_iter == max_iter && error("t=$t resnorm=$resnorm")
    end
    return q
end

function solve_corrector_problem(cache::LocalBackwardEulerCache, model::LocalODEModel)
    @unpack t, dt = cache
    u     = cache.sol.u
    q     = q_to_local_form(cache.sol.q, model)

    J    = local_∂q(u, q, model, t+dt)
    ∂L∂Q = (one(J) - (dt .* J))
    ∂L∂U = -dt * local_∂u(u, q, model, t+dt)
    return ∂L∂Q \ -∂L∂U # = dQdU
end

function assemble_cell!(ke, fe, sol::QuasiStaticSolutionState, cell, cellvalues, material::LinearViscoelasticity1D, stage_cache::AbstractStageCache)
    # @unpack cellvalues = stage_cache
    @unpack u, q = sol
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack E₁   = material.viscosity
    @unpack E₀   = material
    # Iterator info
    for q_point in 1:getnquadpoints(cellvalues)
        # Still iterator info
        info      = QuadraturePointStageInfo(q_point, cellvalues)
        # Common information
        E  = function_gradient(cellvalues, q_point, u)[1] # Frozen u for inner function - dereference here because of 1D function
        # 1. Solve for Qₙ₊₁
        Eᵥref     = q[(q_offset+q_point):(q_offset+q_point+1-1)] # Get current guess for Qₙ₊₁
        local_sol = QuasiStaticSolutionState(E, Eᵥref) # ,,u and q'' for the local problem after another chain rule application
        cache     = build_local_cache(local_sol, cell, info, stage_cache)
        Eᵥ        = solve_local_problem(cache, material.viscosity)
        # 2. Contribution to corrector part ,,dQdU`` by solving the linear system
        dQdE = solve_corrector_problem(cache, material.viscosity)
        # 3. ∂G∂Q
        ∂G∂Q = global_∂q(E, Eᵥ, material, cache)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)[1]
            Tₑ = (E₀ + E₁) * ∇Nᵢ
            Tᵥ = ∂G∂Q * dQdE * ∇Nᵢ
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_gradient(cellvalues, q_point, j)[1]
                ke[i, j] += (∇ˢʸᵐNⱼ * T) * dΩ
                # Corrector part
                ke[i, j] += (∇ˢʸᵐNⱼ * Tᵥ) * dΩ
            end
        end
    end
    return nothing
end

# Standard assembly loop for global problem
function stress!(J, residual, u, q, cache::ElementCache, stage_cache::AbstractStageCache)
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
        assemble_cell!(ke, fe, QuasiStaticSolutionState(u, q), cell, cache.cv, cache.material, stage_cache)
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
            tₚ = cache.material.traction(x, stage_cache.t+stage_cache.dt)
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

local_rhs = viscosity_evolution
local_∂u  = viscosity_evolution_Ju
local_∂q  = viscosity_evolution_Jq

# Smart unpacking of the materials
global_∂q(E,Eᵥ,model::LinearViscoelasticity1D,cache) = global_∂q(u,q,model.viscosity,cache)
global_∂q(E,Eᵥ,model::LinearViscosity1D,cache) = -model.E₁

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
uprev = copy(u₀)
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

        #Setup Jacobian for global problem
        local_tol = outer_iter == 1 ? local_tol⁰ : min(local_tol⁰, norm(Δu)^2)
        local_solver = BackwardEulerStage(t, dt, QuasiStaticSolutionState(uprev,qprev), local_tol, max_iter)
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
    uprev .= u

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
