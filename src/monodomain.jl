
using Ferrite, SparseArrays, BlockArrays
# Instead of using a self written time integrator,
# we will use in this example a time integrator of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
# [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) is a powerful package, from which we will use
# adaptive time stepping. Besides this, almost any ODE solver you can imagine is available.
# In order to use it, we first need to `import` it.
import DifferentialEquations
#
# Now, we define the computational domain and cellvalues. We exploit the fact that all fields of
# the Bidomain model are approximated with the same Ansatz. Hence, we use one CellScalarValues struct for all three fields.
grid = generate_grid(QuadraticQuadrilateral, (30, 30), Vec{2}((0.0,0.0)), Vec{2}((1.0, 1.0)))
dim = 2
Δt = 0.2
T = 10
ip_geo = Lagrange{dim, RefCube, 2}()
ip = Lagrange{dim, RefCube, 2}()
ip_gq = DiscontinuousLagrange{dim, RefCube, 2}() #TODO this is a quick hack to get the right number of dofs into the system
qr = QuadratureRule{dim, RefCube}(3)
cellvalues = CellScalarValues(qr, ip, ip_geo)
# cellvalues_s = CellVectorValues(qr, ip);
#
# We need to intialize a DofHandler. The DofHandler needs to be aware of three different fields
# which are all first order approximations. After pushing all fields into the DofHandler, we `close`
# it and thereby distribute the dofs of the problem.
dh = DofHandler(grid)
push!(dh, :ϕₘ, 1, ip)
push!(dh, :s, 6, ip_gq)
close!(dh);
#
# The linear parts of the Bidomain equations contribute to the stiffness and mass matrix, respectively.
# So, we create a sparsity pattern for those terms.
K = create_sparsity_pattern(dh)
M = create_sparsity_pattern(dh);
#
# Material related parameters are stored in the struct `FHNParameters`
# Base.@kwdef struct FHNParameters
#     a::Float64 = 0.1
#     b::Float64 = 0.5
#     c::Float64 = 1.0
#     d::Float64 = 0.0
#     e::Float64 = 0.01
# end;
#
# Within the equations of the model, spatial dependent parameters occur such as κₑ, κᵢ, Cₘ and χ.
# For the sake of simplicity we kept them constant.
# Nonetheless, we show how one can model spatial dependent coefficients. Hence, the unused function argument `x`
function κ(x)
    return SymmetricTensor{2,2,Float64}((1.4, 0, 1.4/4.0))
end;

function Cₘ(x)
    return 1.0
end;

function χ(x)
    return 1400.0
end;
#
# Boundary conditions are added to the problem in the usual way.
# Please check out the other examples for an in depth explanation.
# Here we force the extracellular porential to be zero at the boundary.
ch = ConstraintHandler(dh)
close!(ch)
update!(ch, 0.0);
#
# We first write a helper to assemble the linear parts. Note that we can precompute and cache linear parts. In the used notation subscripts indicate dependent coefficients.
#
# ```math
# \mathcal{M}
# =
# \begin{bmatrix}
#   M_{\chi C_\textrm{m}} & 0 & 0 \\
#   0 & 0 & 0 \\
#   0 & 0 & M
# \end{bmatrix}
# \qquad
# \mathcal{L}
# =
# \begin{bmatrix}
#   -M_{a\chi}-K_{\bm{\kappa}_{\textrm{i}}} & -K_{\bm{\kappa}_{\textrm{i}}} & -M_{\chi} \\
#   -K_{\bm{\kappa}_{\textrm{i}}} & -K_{\bm{\kappa}_{\textrm{i}}+\bm{\kappa}_{\textrm{e}}} & 0 \\
#   M_{be} & 0 & -M_{bc}
# \end{bmatrix}
# ```
#
# In the following function, `doassemble_linear!`, we assemble all linear parts of the system that stay same over all time steps.
# This follows from the used Method of Lines, where we first discretize in space and afterwards in time.
function doassemble_linear!(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, M::SparseMatrixCSC, dh::DofHandler) where {dim}
    n_ϕₘ = getnbasefunctions(cellvalues)
    n_s = 6*getnquadpoints(cellvalues)
    ntotal = n_ϕₘ + n_s
    n_basefuncs = getnbasefunctions(cellvalues)
    #We use PseudoBlockArrays to write into the right places of Ke
    Ke = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_s], [n_ϕₘ, n_s])
    Me = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_s], [n_ϕₘ, n_s])

    assembler_K = start_assemble(K)
    assembler_M = start_assemble(M)

    #Here the block indices of the variables are defined.
    ϕₘ▄, s▄ = 1, 2

    #Now we iterate over all cells of the grid
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(Me, 0)
        #get the coordinates of the current cell
        coords = getcoordinates(cell)

        Ferrite.reinit!(cellvalues, cell)
        #loop over all Gauss points
        for q_point in 1:getnquadpoints(cellvalues)
            #get the spatial coordinates of the current gauss point
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            #based on the gauss point coordinates, we get the spatial dependent
            #material parameters
            κ_loc = κ(coords_qp)
            Cₘ_loc = Cₘ(coords_qp)
            χ_loc = χ(coords_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            next_i = 1
            for i in 1:n_basefuncs
                Nᵢ = shape_value(cellvalues, q_point, i)
                ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    Nⱼ = shape_value(cellvalues, q_point, j)
                    ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                    #diffusion parts
                    Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] -= ((κ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    #mass matrices
                    Me[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += Cₘ_loc * χ_loc * Nᵢ * Nⱼ * dΩ
                    #for d ∈ 1:6
                        #Me[BlockIndex((s▄,s▄),((i-1)*6+d,(j-1)*6+d))] += Nᵢ * Nⱼ * dΩ
                    #end
                end
            end
            for d ∈ 1:6
                Me[BlockIndex((s▄,s▄),(6*(q_point-1)+d,6*(q_point-1)+d))] = 1.0
            end
        end

        assemble!(assembler_K, celldofs(cell), Ke)
        assemble!(assembler_M, celldofs(cell), Me)
    end
    return K, M
end;

# Regarding the non-linear parts, while the affine term could be cached, for the sake of simplicity we simply recompute it in each call to the right hand side of the system.
# ```math
# \mathcal{N}(
#   \tilde{\varphi}_\textrm{m},
#   \tilde{\varphi}_\textrm{e},
#   \tilde{s})
# =
# \begin{bmatrix}
#   -(\int_\Omega \chi ((\sum_i -\tilde{\varphi}_{m,i} u_{1,i})^3 + \tilde{\varphi}_{m,i} (1+a) u_{1,i})^2)v_{1,j} \textrm{d}\Omega)_j \\
#   0 \\
#   (\int_\Omega de v_{3,j} \textrm{d}\Omega)_j
# \end{bmatrix}
# ```
# It is important to note, that we have to sneak in the boundary conditions into the evaluation of the non-linear term.
#
# TODO cleanup
# The function `apply_nonlinear!` describes the nonlinear change of the system.
# It takes the change vector `du`, the current available solution `u`, the generic storage
# vector `p` and the current time `t`. The storage vector will be used to pass the `dh::DofHandler`,
# `ch::ConstraintHandler`, stiffness matrix `K` and constant material parameters `FHNParameters()`
function apply_nonlinear!(du, u, p, t)
    dh = p[2]
    ch = p[3]
    ip = p[5]
    qr = p[6]
    cellvalues = p[7]
    n_basefuncs = getnbasefunctions(cellvalues)

    du_cell = deepcopy(du[1:7])
    sval = deepcopy(u[1:6])

    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        s_celldofs = _celldofs[dof_range(dh, :s)]
        ϕₘe = u[ϕₘ_celldofs]
        se = u[s_celldofs]
        coords = getcoordinates(cell)
        for q_point in 1:getnquadpoints(cellvalues)
            x_qp = spatial_coordinate(cellvalues, q_point, coords)
            χ_loc = χ(x_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            φₘval = function_value(cellvalues, q_point, ϕₘe)
            # Extract value at quadrature point
            #TODO wrap into function call
            for d in 1:6
                sval[d] = se[6*(q_point-1) + d]
            end
            du_cell .= 0.0            
            if norm(x_qp) < 0.1
                pcg2019_rhs!(du_cell, [φₘval;sval], [t->-maximum([100.0*(1.0-0.5*t), 0.0])], t)
            else
                pcg2019_rhs!(du_cell, [φₘval;sval], [t->0.0], t)
            end
            next_index = 1
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, q_point, j)
                du[ϕₘ_celldofs[j]] += χ_loc * du_cell[1] * Nⱼ * dΩ
            end
            for d in 1:6
                du[s_celldofs[6*(q_point-1) + d]] = du_cell[1+d]
            end
        end
    end
    #apply_zero!(du, ch)
end;
#
# We assemble the linear parts into `K` and `M`, respectively.
K, M = doassemble_linear!(cellvalues, K, M, dh);
M_lumped = sum(M, dims=1);

# Now we apply *once* the boundary conditions to these parts.
apply!(K, ch)
apply!(M, ch);
#
# In the function `monodomain!` we model the actual time dependent DAE problem. This function takes
# the same parameters as `apply_nonlinear!`, which is essentially the defined interface by
# [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
function monodomain!(du, u, p, t)
    K = p[1]
    du .= K * u
    apply_nonlinear!(du, u, p, t)
end;

function monodomain_lumped!(du, u, p, t)
    K = p[1]
    M_lumped = p[4]
    du .= K * u
    apply_nonlinear!(du, u, p, t)
    for i ∈ 1:length(du)
        du[i] /= M_lumped[i]
    end
end;

function monodomain_solve_M!(du, u, p, t)
    K = p[1]
    M = p[8]
    du .= K * u
    apply_nonlinear!(du, u, p, t)
    du .= M \ du
end;
# In the following code block we define the initial condition of the problem. We first
# initialize a zero vector of length `ndofs(dh)` and fill it afterwards in a for loop over all cells.
u₀ = zeros(ndofs(dh));
for cell in CellIterator(dh)
    _celldofs = celldofs(cell)
    ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
    s_celldofs = _celldofs[dof_range(dh, :s)]
    pcg_initial = pcg2019_u₀()
    for i ∈ 1:getnbasefunctions(cellvalues)
        u₀[ϕₘ_celldofs[i]] = pcg_initial[1]
    end
    next_index = 1
    for i ∈ 1:getnquadpoints(cellvalues)
        for d ∈ 1:6
            u₀[s_celldofs[next_index]] = pcg_initial[1+d]
            next_index += 1
        end
    end
end

# We can now state and solve the `ODEProblem`. Since the jacobian of our problem is large and sparse it is advantageous to avoid building a dense matrix (with dense solver) where possible. In [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) we can enforce using sparse jacobian matrices by providing a prototype jacobian with proper sparsity pattern, see [here](https://diffeq.sciml.ai/stable/tutorials/advanced_ode_example/#Speeding-Up-Jacobian-Calculations) for details. In our problem it turns out that the K captures this pattern sufficiently, so for the sake if simplicity we simply use it in this example.
jac_sparsity = sparse(K) + sparse(M)

# f = DifferentialEquations.ODEFunction(monodomain!,mass_matrix=M;jac_prototype=jac_sparsity)
# p = [K, dh, ch, M_lumped, ip, qr, cellvalues]
# prob_mm = DifferentialEquations.ODEProblem(f,u₀,(0.0,T),p)
# sol = DifferentialEquations.solve(prob_mm,DifferentialEquations.Rodas4P(),reltol=1e-4,abstol=1e-5, progress=true, progress_steps = 1, adaptive=true, dt=Δt);

#
# We instantiate a paraview collection file.
# pvd = paraview_collection("monodomain.pvd");
# # Now, we loop over all timesteps and solution vectors, in order to append them to the paraview collection.
# for (solution,t) in zip(sol.u, sol.t)
#     #compress=false flag because otherwise each vtk file will be stored in memory
#     vtk_grid("monodomain-$t.vtu", dh; compress=false) do vtk
#         vtk_point_data(vtk,dh,solution)
#         vtk_save(vtk)
#         pvd[t] = vtk
#     end
# end

p = [K, dh, ch, M_lumped, ip, qr, cellvalues, M]
prob_lumped = DifferentialEquations.ODEProblem(DifferentialEquations.ODEFunction(monodomain_lumped!),u₀,(0.0,T),p)
# prob_explicit_M = DifferentialEquations.ODEProblem(DifferentialEquations.ODEFunction(monodomain_solve_M!),u₀,(0.0,T),p)
sol_explicit = DifferentialEquations.solve(prob_lumped,DifferentialEquations.ROCK4(),reltol=1e-3,abstol=1e-4, progress=true, progress_steps = 1, adaptive=false, dt=Δt);
pvd = paraview_collection("monodomain-explicit-ROCK4.pvd");
# Now, we loop over all timesteps and solution vectors, in order to append them to the paraview collection.
for (solution,t) in zip(sol_explicit.u, sol_explicit.t)
    #compress=false flag because otherwise each vtk file will be stored in memory
    vtk_grid("monodomain-$t-explicit-ROCK4.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,solution)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end

# sol_explicit_ref = DifferentialEquations.solve(prob_lumped,DifferentialEquations.ROCK4(),reltol=1e-7,abstol=1e-8, progress=true, progress_steps = 1, adaptive=true, dt=Δt);
# pvd = paraview_collection("monodomain-explicit-ROCK4-ref.pvd");
# # Now, we loop over all timesteps and solution vectors, in order to append them to the paraview collection.
# for (solution,t) in zip(sol_explicit_ref.u, sol_explicit_ref.t)
#     #compress=false flag because otherwise each vtk file will be stored in memory
#     vtk_grid("monodomain-$t-explicit-ROCK4-ref.vtu", dh; compress=false) do vtk
#         vtk_point_data(vtk,dh,solution)
#         vtk_save(vtk)
#         pvd[t] = vtk
#     end
# end
# Finally, we save the paraview collection.
vtk_save(pvd);
#md # ## [Plain Program](@id bidomain-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [bidomain.jl](bidomain.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
