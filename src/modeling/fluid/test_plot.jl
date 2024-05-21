function p_LV(t,p)
    t_periodic = mod(t,p.THB)/p.THB
    if t_periodic < 0.75
        return 10.0 #+ 100*sin(π*t_periodic/0.75)^2
    else
        return 10.0
    end
end

function test_solve_lumped()
    p = Thunderbolt.RSAFDQ2022LumpedCicuitModel{Float64,Float64,Float64,Float64,Float64,Float64}(;lv_pressure_given=true)

    # V_LV(t) = 140 + -20*(-abs(2*(mod(t-0.5*p.THB,p.THB))/p.THB)+1)

    Δt = 0.001#1.0
    # τ = 0.0:Δt:(p.THB-Δt)
    τ = 0.0:Δt:(100*p.THB-Δt)
    # τ = 0.0:Δt:(0.275-Δt)
    solutions = zeros(Thunderbolt.num_states(p), length(τ)+1)
    u₀ = @view solutions[:, 1]
    Thunderbolt.default_initial_condition!(u₀, p)
    # solutions[2, 1] = 140.0
    du = zeros(Thunderbolt.num_states(p))
    for (i,t) ∈ enumerate(τ)
        # solutions[2, i] = V_LV(t)
        solutions[:, i+1] .= solutions[:, i]
        Thunderbolt.lumped_driver!(du, solutions[:, i], t, [p_LV(t,p)], p)
        solutions[:, i+1] += Δt * du
    end
    return solutions
end


function plot_solution(solution)
    p = Thunderbolt.RSAFDQ2022LumpedCicuitModel{Float64,Float64,Float64,Float64,Float64,Float64}()

    Δt = 0.001# 1.0
    τ = collect(0.0:Δt:100*p.THB)
    # τ = collect(0.0:Δt:p.THB)
    # τ = 0.0:Δt:(0.275)

    # V_LV(t) = 80 + -(-abs(2*(mod(t-0.5*p.THB,p.THB))/p.THB)+1)

    @inline Eₗᵥ(p,t) = Thunderbolt.elastance_RSAFDQ2022(t, p.Epassᵣₐ, 10*p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eₗₐ(p,t) = Thunderbolt.elastance_RSAFDQ2022(t, p.Epassₗₐ, p.Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    @inline Eᵣₐ(p,t) = Thunderbolt.elastance_RSAFDQ2022(t, p.Epassᵣₐ, p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eᵣᵥ(p,t) = Thunderbolt.elastance_RSAFDQ2022(t, p.Epassᵣᵥ, p.Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)

    Vₗₐ = @view solution[1,:]
    Vₗᵥ = @view solution[2,:]
    Vᵣₐ = @view solution[3,:]
    Vᵣᵥ = @view solution[4,:]
    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = p
    pₗᵥ = [Eₗᵥ(p, τ[i])*(Vₗᵥ[i] - 5.0) for i ∈ 1:length(τ)]
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

    lines!(axs[1], Vₗᵥ, pₗₐ) # V_LV.(τ))
    lines!(axs[2], Vᵣᵥ, pᵣᵥ)
    lines!(axs[3], Vₗₐ, pₗₐ)
    lines!(axs[4], Vᵣₐ, pᵣₐ)
    f

    # f = Figure()
    # axs = [
    #     Axis(f[1, 1], title="V"),
    #     Axis(f[1, 2], title="p"),
    #     Axis(f[1, 3], title="dp"),
    #     Axis(f[1, 4], title="Q"),
    # ]

    # lines!(axs[1], τ, Vₗᵥ, label="lv") # V_LV.(τ))
    # lines!(axs[1], τ, Vᵣᵥ, label="rv")
    # lines!(axs[1], τ, Vₗₐ, label="la", linestyle = :dash)
    # lines!(axs[1], τ, Vᵣₐ, label="ra")
    # axislegend(axs[1])

    # lines!(axs[2], τ, [p_LV(t,p) for t in τ]) # V_LV.(τ))
    # lines!(axs[2], τ, pᵣᵥ)
    # lines!(axs[2], τ, pₗₐ, linestyle = :dash)
    # lines!(axs[2], τ, pᵣₐ)

    # lines!(axs[3], τ, solution[5,:], label="lv") # V_LV.(τ))
    # lines!(axs[3], τ, solution[6,:], label="rv")
    # lines!(axs[3], τ, solution[7,:], label="la", linestyle = :dash)
    # lines!(axs[3], τ, solution[8,:], label="ra")

    # lines!(axs[4], τ, solution[9,:], label="lv") # V_LV.(τ))
    # lines!(axs[4], τ, solution[10,:], label="rv")
    # lines!(axs[4], τ, solution[11,:], label="la", linestyle = :dash)
    # lines!(axs[4], τ, solution[12,:], label="ra")

    # f
end
