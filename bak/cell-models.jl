sigmoid(φ, E_Y, k_Y, sign) = 1.0 / (1.0 + exp(sign * (φ - E_Y) / k_Y))

function pcg2019_rhs_fast!(du,u,p,t)
     I_stim = p[1]
     g_to = p[2]
     g_CaL = p[3]
     E_d = p[4]
     E_f = p[5]
     g_K1 = p[6]

    φ  = u[1]
    h  = u[2]
    m  = u[3]
    f  = u[4]
    s  = u[5]
    xs = u[6]
    xr = u[7]

    C_m = 1.0 # [µF/cm^-2]
    # ------ I_Na -------
    g_Na = 12.0    # [mS/µF]
    E_m  = -52.244 # [mV]
    k_m  = 6.5472  # [mV]
    τ_m  = 0.12    # [ms]
    E_h  = -78.7   # [mV]
    k_h  = 5.93    # [mV]
    δ_h  = 0.799163 # dimensionless
    τ_h0 = 6.80738  # [ms]
    # ------ I_K1 -------
    #g_K1 = 0.73893  # [mS/µF]
    E_z  = -91.9655 # [mV]
    k_z  = 12.4997  # [mV]
    # ------ I_to -------
    #g_to = 0.1688   # [mS/µF]
    E_r  = 14.3116  # [mV]
    k_r  = 11.462   # [mV]
    E_s  = -47.9286 # [mV]
    k_s  = 4.9314   # [mV]
    τ_s  = 9.90669  # [ms]
    # ------ I_CaL -------
    #g_CaL = 0.11503 # [mS/µF]
    #E_d   = 0.7     # [mV]
    k_d   = 4.3     # [mV]
    #E_f   = -15.7   # [mV]
    k_f   = 4.6     # [mV]
    τ_f   = 30.0    # [ms]
    # ------ I_Kr -------
    g_Kr = 0.056 # [mS/µF]
    E_xr = -26.6 # [mV]
    k_xr = 6.5   # [mV]
    τ_xr = 334.0 # [ms]
    E_y  = -49.6 # [mV]
    k_y  = 23.5  # [mV]
    # ------- I_Ks --------
    g_Ks = 0.008 # [mS/µF]
    E_xs = 24.6  # [mV]
    k_xs = 12.1  # [mV]
    τ_xs = 628.0 # [ms]
    # ------- Other --------
    E_Na = 65.0  # [mV]
    E_K  = -85.0 # [mV]
    E_Ca = 50.0  # [mV]

    # Instantaneous gates
    r∞ = sigmoid(φ, E_r, k_r, -1.0)
    d∞ = sigmoid(φ, E_d, k_d, -1.0)
    z∞ = sigmoid(φ, E_z, k_z, 1.0)
    y∞ = sigmoid(φ, E_y, k_y, 1.0)

    # Currents
    I_Na  = g_Na * m * m * m * h * h * (φ - E_Na)
    I_K1  = g_K1 * z∞ * (φ - E_K)
    I_to  = g_to * r∞ * s * (φ - E_K)
    I_CaL = g_CaL * d∞ * f * (φ - E_Ca)
    I_Kr  = g_Kr * xr * y∞ * (φ - E_K)
    I_Ks  = g_Ks * xs * (φ - E_K)

    I_total = I_Na + I_K1 + I_to + I_CaL + I_Kr + I_Ks + I_stim(t)

    du[1] = -I_total/C_m

    τ_h = (2.0 * τ_h0 * exp(δ_h * (φ - E_h) / k_h)) / (1.0 + exp((φ - E_h) / k_h))
    h∞ = sigmoid(φ, E_h, k_h, 1.0)
    du[2] = (h∞-h)/τ_h

    m∞ = sigmoid(φ, E_m, k_m, -1.0)
    du[3] = (m∞-m)/τ_m
end

function pcg2019_rhs_slow!(du,u,p,t)
     g_to = p[2]
     g_CaL = p[3]
     E_d = p[4]
     E_f = p[5]
     g_K1 = p[6]

    φ  = u[1]
    f  = u[4]
    s  = u[5]
    xs = u[6]
    xr = u[7]

    #float C_m = 0.01 # [µF/mm^-2]
    # ------ I_Na -------
    g_Na = 12.0    # [mS/µF]
    E_m  = -52.244 # [mV]
    k_m  = 6.5472  # [mV]
    τ_m  = 0.12    # [ms]
    E_h  = -78.7   # [mV]
    k_h  = 5.93    # [mV]
    δ_h  = 0.799163 # dimensionless
    τ_h0 = 6.80738  # [ms]
    # ------ I_K1 -------
    #g_K1 = 0.73893  # [mS/µF]
    E_z  = -91.9655 # [mV]
    k_z  = 12.4997  # [mV]
    # ------ I_to -------
    #g_to = 0.1688   # [mS/µF]
    E_r  = 14.3116  # [mV]
    k_r  = 11.462   # [mV]
    E_s  = -47.9286 # [mV]
    k_s  = 4.9314   # [mV]
    τ_s  = 9.90669  # [ms]
    # ------ I_CaL -------
    #g_CaL = 0.11503 # [mS/µF]
    #E_d   = 0.7     # [mV]
    k_d   = 4.3     # [mV]
    #E_f   = -15.7   # [mV]
    k_f   = 4.6     # [mV]
    τ_f   = 30.0    # [ms]
    # ------ I_Kr -------
    g_Kr = 0.056 # [mS/µF]
    E_xr = -26.6 # [mV]
    k_xr = 6.5   # [mV]
    τ_xr = 334.0 # [ms]
    E_y  = -49.6 # [mV]
    k_y  = 23.5  # [mV]
    # ------- I_Ks --------
    g_Ks = 0.008 # [mS/µF]
    E_xs = 24.6  # [mV]
    k_xs = 12.1  # [mV]
    τ_xs = 628.0 # [ms]
    # ------- Other --------
    E_Na = 65.0  # [mV]
    E_K  = -85.0 # [mV]
    E_Ca = 50.0  # [mV]

    f∞ = sigmoid(φ, E_f, k_f, 1.0)
    du[4] = (f∞-f)/τ_f

    s∞ = sigmoid(φ, E_s, k_s, 1.0)
    du[5] = (s∞-s)/τ_s

    xs∞ = sigmoid(φ, E_xs, k_xs, -1.0)
    du[6] = (xs∞-xs)/τ_xs

    xr∞ = sigmoid(φ, E_xr, k_xr, -1.0)
    du[7] = (xr∞-xr)/τ_xr
end

function pcg2019_rhs!(du,u,p,t)
    pcg2019_rhs_fast!(du,u,p,t)
    pcg2019_rhs_slow!(du,u,p,t)
end

function pcg2019_u₀(p)
     g_to = p[2]
     g_CaL = p[3]
     E_d = p[4]
     E_f = p[5]
     g_K1 = p[6]

    g_Na = 12.0    # [mS/µF]
    E_m  = -52.244 # [mV]
    k_m  = 6.5472  # [mV]
    τ_m  = 0.12    # [ms]
    E_h  = -78.7   # [mV]
    k_h  = 5.93    # [mV]
    δ_h  = 0.799163 # dimensionless
    τ_h0 = 6.80738  # [ms]
    # ------ I_K1 -------
    #g_K1 = 0.73893  # [mS/µF]
    E_z  = -91.9655 # [mV]
    k_z  = 12.4997  # [mV]
    # ------ I_to -------
    #g_to = 0.1688   # [mS/µF]
    E_r  = 14.3116  # [mV]
    k_r  = 11.462   # [mV]
    E_s  = -47.9286 # [mV]
    k_s  = 4.9314   # [mV]
    τ_s  = 9.90669  # [ms]
    # ------ I_CaL -------
    #g_CaL = 0.11503 # [mS/µF]
    #E_d   = 0.7     # [mV]
    k_d   = 4.3     # [mV]
    #E_f   = -15.7   # [mV]
    k_f   = 4.6     # [mV]
    τ_f   = 30.0    # [ms]
    # ------ I_Kr -------
    g_Kr = 0.056 # [mS/µF]
    E_xr = -26.6 # [mV]
    k_xr = 6.5   # [mV]
    τ_xr = 334.0 # [ms]
    E_y  = -49.6 # [mV]
    k_y  = 23.5  # [mV]
    # ------- I_Ks --------
    g_Ks = 0.008 # [mS/µF]
    E_xs = 24.6  # [mV]
    k_xs = 12.1  # [mV]
    τ_xs = 628.0 # [ms]
    # ------- Other --------
    E_Na = 65.0  # [mV]
    E_K  = -85.0 # [mV]
    E_Ca = 50.0  # [mV]

    u₀ = zeros(7)
    u₀[1] = E_K
    u₀[2] = sigmoid(u₀[1], E_h, k_h, 1.0)
    u₀[3] = sigmoid(u₀[1], E_m, k_m, -1.0)
    u₀[4] = sigmoid(u₀[1], E_f, k_f, 1.0)
    u₀[5] = sigmoid(u₀[1], E_s, k_s, 1.0)
    u₀[6] = sigmoid(u₀[1], E_xs, k_xs, -1.0)
    u₀[7] = sigmoid(u₀[1], E_xr, k_xr, -1.0)
    return u₀
end

function empty_rhs!(du,u,p,t)
    du .= 0.0
end

using DifferentialEquations
using Plots

#params = [t->-maximum([50.0*(1.0-0.5*mod(t,400.0)), 0.0]), 0.1688*1.64, 0.11503/1.1, 0.7/1.2, -15.7*1.1, 0.73893/1.2]
params = [t->-maximum([50.0*(1.0-0.5*mod(t,450.0)), 0.0]), 0.1688*1.9, 0.11503, 0.7, -15.7, 0.73893]
u₀ = pcg2019_u₀(params)
tspan = (0.0,5000.0)
prob = ODEProblem(pcg2019_rhs!,u₀,tspan,params)
sol = solve(prob, Tsit5())

# prob_split = SplitODEProblem(pcg2019_rhs_fast!, pcg2019_rhs_slow!, u₀, tspan)
# sol_split = solve(prob_split, SBDF4(), dt=0.1)

# sol_split = solve(prob, ESERK5(), dt=0.25, adaptive=false)
# sol_split = solve(prob, ROCK4(), dt=0.2, adaptive=true)
# sol_split = solve(prob, Rodas4P(), dt=0.2, adaptive=false)
# sol_split = solve(prob, PFRK87(), dt=0.4, adaptive=false)
# sol_split = solve(prob, SSPRK104(), dt=0.5, adaptive=false)
# sol_split = solve(prob, SSPRK43(), dt=0.2, adaptive=false)
# sol_split = solve(prob, SSPRK53(), dt=0.3, adaptive=false)
# sol_split = solve(prob, GRK4T(), dt=0.25, adaptive=false)
# sol_split = solve(prob, RosenbrockW6S4OS(), dt=0.25, adaptive=false)

# prob_split_exp = SplitODEProblem(empty_rhs!, pcg2019_rhs!, u₀, tspan)
# sol_split = solve(prob_split_exp, HochOst4(), dt=0.25, adaptive=false)

φplot = plot(sol,linewidth=2,title="PCG2019",
     xaxis="Time (ms)",yaxis="",label="Reference Solution",vars=[1])
# φplot = plot!(sol_split,linewidth=2,title="PCG2019",
#      xaxis="Time (ms)",yaxis="",label="SBDF4",vars=[1])

splot = plot(sol,linewidth=2,title="PCG2019",
     xaxis="Time (ms)",yaxis="",label="h ref",vars=[2])
splot = plot!(sol,linewidth=2,title="PCG2019",
     xaxis="Time (ms)",yaxis="",label="m ref",vars=[3])
# splot = plot!(sol_split,linewidth=2,title="PCG2019",
#      xaxis="Time (ms)",yaxis="",label="h SBDF4",vars=[2])
# splot = plot!(sol_split,linewidth=2,title="PCG2019",
     # xaxis="Time (ms)",yaxis="",label="m SBDF4",vars=[3])

plot(φplot, splot, layout=(1,2))
