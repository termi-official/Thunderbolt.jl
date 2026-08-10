using Thunderbolt
import SciMLBase
using Test
using LinearAlgebra
using LinearSolve

# Observed order of convergence in time. A scheme that forms its rate from the wrong `Δt` still
# *converges* -- to the wrong limit -- so these assert on the measured order rather than merely on the
# differences shrinking.
#
# The drives are smooth in time on purpose: a rate discontinuity in the data destroys the observed
# order whatever the scheme does, so measuring across one would say nothing about the discretization.

const ORTHO_MS = Thunderbolt.ConstantCoefficient(
    Thunderbolt.OrthotropicMicrostructure(
        Vec((1.0, 0.0, 0.0)),
        Vec((0.0, 1.0, 0.0)),
        Vec((0.0, 0.0, 1.0)),
    ),
)

"""
Ratios of successive solution differences under repeated step halving. A scheme of order `p` sends
these to `2^p`.

The differences are norms of the *difference vectors*, not differences of norms: `|‖u₁‖ - ‖u₂‖|` can
collapse when the error changes sign across components, and the ratio would then measure a
cancellation rather than the order.
"""
function convergence_ratios(solve_to_end, Δts)
    us = [solve_to_end(Δt) for Δt in Δts]
    diffs = [norm(us[i+1] .- us[i]) for i = 1:(length(us)-1)]
    return [diffs[i] / diffs[i+1] for i = 1:(length(diffs)-1)]
end

@testset "Viscous Robin is first order in time" begin
    # The dt -> 0 limit is what says the velocity reaching a facet is the one that actually separates
    # `uₙ` from `uₙ₋₁`.
    mesh = generate_mesh(Hexahedron, (2, 2, 2))
    material = Thunderbolt.PK1Model(Guccione1991PassiveModel(), ORTHO_MS)

    function solve_to_one(facet_models, Δt)
        dbcs = [
            Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
            Dirichlet(
                :d,
                getfacetset(mesh, "right"),
                (x, t) -> (0.1 * sinpi(2t), 0.0, 0.0),
                [1, 2, 3],
            ),
        ]
        form = semidiscretize(
            QuasiStaticModel(:d, material, facet_models),
            FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
            mesh,
        )
        integrator = init(
            QuasiStaticProblem(form, (0.0, 1.0)),
            BackwardEulerSolver(
                inner_solver = NewtonRaphsonSolver(
                    max_iter = 20,
                    tol = 1e-10,
                    inner_solver = UMFPACKFactorization(),
                ),
            ),
            dt = Δt,
            verbose = false,
        )
        solve!(integrator)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        return copy(integrator.u)
    end

    Δts = (0.05, 0.025, 0.0125, 0.00625)
    ratios = convergence_ratios(Δt -> solve_to_one((ViscousRobinBC(1.0, "top"),), Δt), Δts)
    # Backward Euler is first order, so halving the step halves the increment.
    @test all(r -> 1.7 ≤ r ≤ 2.3, ratios)

    # The drive returns to the undeformed configuration at t = 1, so without a dashpot the
    # quasi-static answer there is exactly zero -- at any step size, so the cheapest one will do. The
    # dashpot's lag is what makes it non-zero, and that lag *survives* refinement rather than being a
    # coarse-step artefact. Those are the two halves of "faithful".
    @test norm(solve_to_one((), 0.5)) < 1e-10
    @test norm(solve_to_one((ViscousRobinBC(1.0, "top"),), minimum(Δts))) > 1e-2
end
