# # [Electrophysiology Tutorial 1: Simple Spiral Wave](@id ep-tutorial_simple-spiral-wave)
# ![Spiral Wave](spiral-wave.gif)
#
# This tutorial shows how to perform a simulation of electrophysiological behavior of cardiac tissue.
#
# ## Introduction
#
# The most widespread model of cardiac electrophysiology is the monodomain model.
# It can be defined on a domain $\Omega$ as the system of partial differential equations
#
# ```math
# \begin{aligned}
#   \chi C_{\textrm{m}} \partial_t \varphi &= \nabla \cdot \boldsymbol{\kappa} \nabla \varphi - \chi I'(\varphi, \boldsymbol{s}, t) & \textrm{in} \: \Omega \, , \\
#   \partial_t \boldsymbol{s} &= \mathbf{g}(\varphi, \boldsymbol{s}) & \textrm{in}  \: \Omega \, , \\
#   0 &= \boldsymbol{\kappa} \nabla \varphi \cdot \mathbf{n} & \mathrm{on} \: \partial \Omega \, ,
# \end{aligned}
# ```
#
# together with admissible initial conditions and a cellular ionic model to determine $I'$ and $\mathbf{g}$. 
# ${\boldsymbol{\kappa}}$ denotes the conductivity tensor, $\varphi$ is the transmembrane potential field, $\chi$ is the volume to membrane surface ratio, $C_{\mathrm{m}}$ is the membrane capacitance, and $I'(\varphi, \boldsymbol{s}, t) := I_{\textrm{ion}}(\varphi, \boldsymbol{s}) + I_{\textrm{stim}}(t)$ denotes the sum of the ionic current due to the cell model and the applied stimulus current, respectively.
#
# In this tutorial we will apply a reaction-diffusion split to this model and solve it with an operator splitting solver.
# For some theory on operator splitting we refer to the [theory manual on operator splitting](@ref theory_operator-splitting).
#
# ## Commented Program
# We start by loading Thunderbolt and LinearSolve to use a custom direct solver of our choice.
using Thunderbolt, LinearSolve

# We start by constructing a square domain for our simulation.
mesh = generate_mesh(Quadrilateral, (2^6, 2^6), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

# We now define the parameters appearing in the model.
# For simplciity we assume $C_{\mathrm{m}} = \chi = 1.0$ and a homogeneous, anisotropic symmetric conductivity tensor.
Cₘ = ConstantCoefficient(1.0)
χ  = ConstantCoefficient(1.0)
κ  = ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)))

# The spiral wave will unfold due to the specific construction of the initial conditions, hence we do not need to apply a stimulus.
stimulation_protocol = NoStimulationProtocol()

# Now we choose a cell model.
# For simplicity we choose a neuronal electrophysiology model, which is a nice playground.
cell_model = Thunderbolt.FHNModel()
# !!! tip
#     A full list of all models can be found in the [API reference](api-reference/models/#Cells).
#     To implement a custom cell model please consult [the how-to section](@ref how-to-custom-ep-cell-model).

# !!! todo
#     The initializer API is not yet finished and hence we deconstruct stuff here manually.
#     Please note that this method is quite fragile w.r.t. to many changes you can make in the code below.
# Spiral wave initializer for the FitzHugh-Nagumo
function spiral_wave_initializer!(u₀, f::GenericSplitFunction)
    ## TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.solution_indices[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    ## TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    ## Should not be reshape but some array of arrays fun, because in general (e.g. for heterogeneous tissues) we cannot reshape into a matrix
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));

    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        φₘ_celldofs = _celldofs[dof_range(dh, :φₘ)]
        ## TODO query coordinate directly from the cell model
        coordinates = getcoordinates(cell)
        for (i, (x₁, x₂)) in zip(φₘ_celldofs,coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                φ₀[i] = 1.0
            end
            if x₂ >= 1.25
                s₀[i,1] = 0.1
            end
        end
    end
end

# Now we put the components together by instantiating the monodomain model.
model = MonodomainModel(
    Cₘ,
    χ,
    microstructure,
    stimulation_protocol,
    cell_model,
    :φₘ, :s,
)


csc = CoordinateSystemCoefficient(
    CartesianCoordinateSystem(mesh)
)

odeform = semidiscretize(
    ReactionDiffusionSplit(model, csc),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)
u₀ = zeros(Float32, OS.function_size(odeform))

spiral_wave_initializer!(u₀, odeform)

u₀gpu = CuVector(u₀)
problem = OS.OperatorSplittingProblem(odeform, u₀gpu, tspan)

heat_timestepper = BackwardEulerSolver(
    solution_vector_type=CuVector{Float32},
    system_matrix_type=CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
    inner_solver=KrylovJL_CG(atol=1.0f-6, rtol=1.0f-5),
)
# !!! tip
#     On non-trivial geometries it is highly recommended to use a preconditioner.
#     Please consult the [LinearSolve.jl docs](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/) for details.

cell_timestepper = AdaptiveForwardEulerSubstepper(
    solution_vector_type=CuVector{Float32},
    reaction_threshold=0.1f0,
)

timestepper = OS.LieTrotterGodunov((heat_timestepper, cell_timestepper))

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

io = ParaViewWriter("spiral-wave-test")

# TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    dh = odeform.functions[1].dh
    φ = u[odeform.dof_ranges[1]]
    @info t,norm(u)
    # sflat = ....?
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, Vector(φ), :φₘ)
        # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
        # for sidx in 1:Thunderbolt.num_states(ionic_model)
        #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
        # end
    end
end
TimerOutputs.print_timer()
# TimerOutputs.disable_debug_timings(Thunderbolt)
