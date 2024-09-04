model = Thunderbolt.RDQ20MFModel()
du = zeros(Thunderbolt.num_states(model))
u = zeros(Thunderbolt.num_states(model))
Thunderbolt.initial_state!(u, model)
# u .= [0.0007975034186979656, 0.0006243448053976429, 1.10736687557628e-6, 0.00012484286758916483, 0.0006243448053976429, 0.0004887917564196871, 0.00012484286758916483, 0.014076353941490436, 0.00537736424408801, 0.004209652025338546, 7.466699041003822e-5, 0.00841784078555284, 0.004209652025338546, 0.00329566942327174, 0.00841784078555284, 0.949135181890976, 0.23422179001163454, 0.005575929820039651, 0.004000763869507678, 9.523042316664843e-5]

dt = 1e-5
Tmax = 0.5

# Calcium transient
const c0 = 0.1
const cmax = 0.9
const τ1 = .02; # [s]
const τ2 = .05; # [s]
const t0 = 0.01;  # [s]
const β = (τ1 / τ2)^(-1 / (τ1 / τ2 - 1)) - (τ1 / τ2)^( -1 / (1 - τ2 / τ1))

function Ca(tin)
    t = tin % 0.5
    t < t0 ? c0 : c0 + ((cmax - c0) / β * (exp(-(t - t0) / τ1) - exp(-(t - t0) / τ2)))
    # 0.1
end

# SL transient
const SL0 = 2.2;       # [micro m]
const SL1 = SL0 * .9; # [micro m]
const SLt0 = .05;      # [s]
const SLt1 = .35;      # [s]
const SLτ0 = .05;    # [s]
const SLτ1 = .02;    # [s]

function Sl(tin)
    t = tin % 0.5
    (SL0 + (SL1 - SL0) * (max(0.0, 1.0 - exp((SLt0 - t) / SLτ0)) - max(0.0, 1.0 - exp((SLt1 - t) / SLτ1))))/SL0
    # 1.0
end

τ = 0.0:dt:Tmax
us = zeros(Thunderbolt.num_states(model), length(τ))
Tas = zeros(length(τ))
for (i,t) ∈ enumerate(τ)
    calcium = Ca(t)
    sarcomere_length = Sl(t)
    sarcomere_velocity = 0.0# (Sl(t+1e-8) - sarcomere_length)/1e-8 # TODO via AD
    Thunderbolt.rhs!(du, u, nothing, sarcomere_length, sarcomere_velocity, calcium, t, model)
    u .+= dt*du
    us[:,i] .= u

    Tas[i] = Thunderbolt.compute_active_tension(model, u, SL0*sarcomere_length)
end

# Reference solution up to a permutation of the indices
# Ca=0.1
# SL=2.2
# SLdT=0.0
uref = [0.74533,0.211469,0.00750457,0.00212916,0.00103517,0.00293704,0.00150072,0.00425779,0.00750457,0.00212916,7.55578e-05,2.14334e-05,0.00150072,0.00425779,0.00217549,0.00617217,0.00155623,3.70791e-05,0.00423817,0.0001009]
@test uref[16:20] ≈ u[16:20] atol=1e-6
