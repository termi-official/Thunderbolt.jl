using Thunderbolt, DelimitedFiles, Test

@testset "RDQ20MFModel" begin
    @testset "Active value trajectories" begin
        datapath = joinpath(@__DIR__, "data", "trajectories", "RDQ20-MF", "transient-test.csv")
        (reference_solution_data, header) = readdlm(datapath, ',', Float64, '\n'; header=true)
        header = header[:]
        tidx = findfirst(i->i=="t", header)
        CAidx = findfirst(i->i=="Ca", header)
        SLidx = findfirst(i->i=="SL", header)
        dSLidx = findfirst(i->i=="dSL_dt", header)
        Taidx = findfirst(i->i=="Ta", header)
        Asidx = findfirst(i->i=="As", header)

        # 1000x to translate from s to ms
        ts_data = 1000.0*reference_solution_data[:,tidx]

        model = Thunderbolt.RDQ20MFModel(;
            calcium_field = ConstantCoefficient(0.0),
            sarcomere_length = ConstantCoefficient(2.2),
            sarcomere_velocity = ConstantCoefficient(0.0)
        )
        du = zeros(Thunderbolt.num_states(model))
        u  = zeros(Thunderbolt.num_states(model))
        u[1] = 1.0

        dt = 1e-2
        Tmax = 600.0

        # Calcium transient
        c0 = 0.1
        cmax = 0.9
        τ1 = 20.0; # ms
        τ2 = 50.0; # ms
        t0 = 10.0;  # ms
        β = (τ1 / τ2)^(-1 / (τ1 / τ2 - 1)) - (τ1 / τ2)^( -1 / (1 - τ2 / τ1))

        Ca(t) = t < t0 ? c0 : c0 + ((cmax - c0) / β * (exp(-(t - t0) / τ1) - exp(-(t - t0) / τ2)))

        # SL transient
        SL0 = 2.2;       # µm
        SL1 = SL0 * .97; # µm
        SLt0 = 50.0;      # ms
        SLt1 = 350.0;      # ms
        SLτ0 = 50.0;    # ms
        SLτ1 = 20.0;    # ms

        Sl(t) = SL0 + (SL1 - SL0) * (max(0.0, 1.0 - exp((SLt0 - t) / SLτ0)) - max(0.0, 1.0 - exp((SLt1 - t) / SLτ1)));

        τ = 0.0:dt:Tmax
        for (i,t) ∈ enumerate(τ)
            calcium = Ca(t)
            sarcomere_length = Sl(t)
            sarcomere_velocity = (sarcomere_length - Sl(t-1e-5))/1e-5 # TODO via AD
            Thunderbolt.rhs!(du, u, Vec((0.0,)), 0.0, Thunderbolt.RDQ20MFModel(;
                calcium_field = ConstantCoefficient(calcium),
                sarcomere_length = ConstantCoefficient(sarcomere_length),
                sarcomere_velocity = ConstantCoefficient(sarcomere_velocity)
            ))
            u .+= dt*du

            closest_sol_idx = findfirst(tref -> t-dt/2 ≤ tref < t+dt/2, ts_data)
            if closest_sol_idx !== nothing
                @test calcium ≈ reference_solution_data[closest_sol_idx, CAidx] rtol=1e-3
                @test sarcomere_velocity ≈ reference_solution_data[closest_sol_idx, dSLidx] rtol=1e-3
                @test sarcomere_length ≈ reference_solution_data[closest_sol_idx, SLidx] rtol=1e-3
                Ta = Thunderbolt.compute_active_tension(model, u, sarcomere_length)
                @test Ta ≈ reference_solution_data[closest_sol_idx, Taidx] rtol=1e-3
                As = Thunderbolt.compute_active_stiffness(model, u, sarcomere_length)
                @test As ≈ reference_solution_data[closest_sol_idx, Asidx] rtol=1e-3
            end
        end
    end
end
