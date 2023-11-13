function test_solve_lumped()
    p = Thunderbolt.ReggazoniSalvadorAfricaLumpedCicuitModel{Float64,Float64,Float64,Float64,Float64}()

    V_LV(t) = 80 + -(-abs(2*(mod(t-0.5*p.THB,p.THB))/p.THB)+1)
    p_LV(t) = 10.0 + (sin(2*π*mod(t,p.THB)/p.THB)+1)

    Δt = 0.001
    τ = 0.0:Δt:(10*p.THB-Δt)
    solutions = zeros(Thunderbolt.num_states(p), length(τ)+1)
    u₀ = @view solutions[:, 1]
    Thunderbolt.initial_condition!(u₀, p)
    du = zeros(Thunderbolt.num_states(p))
    for (i,t) ∈ enumerate(τ)
        solutions[2, i] = V_LV(t)
        solutions[:, i+1] .= solutions[:, i]
        Thunderbolt.lumped_driver_lv!(du, solutions[:, i], t, p_LV(t), p)
        solutions[:, i+1] += Δt * du
    end
    return solutions
end


function plot_solution(solution)
    p = Thunderbolt.ReggazoniSalvadorAfricaLumpedCicuitModel{Float64,Float64,Float64,Float64,Float64}()

    Δt = 0.001
    τ = collect(0.0:Δt:10*p.THB)

    # V_LV(t) = 80 + -(-abs(2*(mod(t-0.5*p.THB,p.THB))/p.THB)+1)
    p_LV(t) = 10.0 + (sin(2*π*mod(t,p.THB)/p.THB)+1)

    @inline Eₗₐ(p,t) = Thunderbolt.elastance_ReggazoniSalvadorAfrica(t, p.Epassₗₐ, p.Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    @inline Eᵣₐ(p,t) = Thunderbolt.elastance_ReggazoniSalvadorAfrica(t, p.Epassᵣₐ, p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eᵣᵥ(p,t) = Thunderbolt.elastance_ReggazoniSalvadorAfrica(t, p.Epassᵣᵥ, p.Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)

    Vₗₐ = @view solution[1,:]
    Vₗᵥ = @view solution[2,:]
    Vᵣₐ = @view solution[3,:]
    Vᵣᵥ = @view solution[4,:]
    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = p
    pₗₐ = [Eₗₐ(p, τ[i])*(Vₗₐ[i] - V0ₗₐ) for i ∈ 1:length(τ)]
    pᵣₐ = [Eᵣₐ(p, τ[i])*(Vᵣₐ[i] - V0ᵣₐ) for i ∈ 1:length(τ)]
    pᵣᵥ = [Eᵣᵥ(p, τ[i])*(Vᵣᵥ[i] - V0ᵣᵥ) for i ∈ 1:length(τ)]
    @show extrema(Vᵣₐ), extrema(pᵣₐ)
    f = Figure()
    axs = [
        Axis(f[1, 1], title="LV"),
        Axis(f[1, 2], title="RV"),
        Axis(f[2, 1], title="LA"),
        Axis(f[2, 2], title="RA")
    ]

    lines!(axs[1], Vₗᵥ, p_LV.(τ)) # V_LV.(τ))
    lines!(axs[2], Vᵣᵥ, pᵣᵥ)
    lines!(axs[3], Vₗₐ, pₗₐ)
    lines!(axs[4], Vᵣₐ, pᵣₐ)
    f
end
