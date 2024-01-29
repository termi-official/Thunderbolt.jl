model = Thunderbolt.RDQ20MFModel(;
    calcium_field = ConstantCoefficient(0.0),
    sarcomere_length = ConstantCoefficient(2.1),
    sarcomere_velocity = ConstantCoefficient(0.0)
)
du = zeros(Thunderbolt.num_states(model))
u = zeros(Thunderbolt.num_states(model))
Thunderbolt.initial_state!(u, model)
@show u


dt = 1e-6
const Tmax = 0.1

# Calcium transient
const c0 = 0.1
const cmax = 0.9
const τ1 = .02; # [s]
const τ2 = .05; # [s]
const t0 = 0.01;  # [s]
const β = (τ1 / τ2)^(-1 / (τ1 / τ2 - 1)) - (τ1 / τ2)^( -1 / (1 - τ2 / τ1))

Ca(t) = t < t0 ? c0 : c0 + ((cmax - c0) / β * (exp(-(t - t0) / τ1) - exp(-(t - t0) / τ2)))

# SL transient
const SL0 = 2.2;       # [micro m]
const SL1 = SL0 * .97; # [micro m]
const SLt0 = .05;      # [s]
const SLt1 = .35;      # [s]
const SLτ0 = .05;    # [s]
const SLτ1 = .02;    # [s]

Sl(t) = SL0 + (SL1 - SL0) * (max(0.0, 1.0 - exp((SLt0 - t) / SLτ0)) - max(0.0, 1.0 - exp((SLt1 - t) / SLτ1)));

τ = 0.0:dt:Tmax
us = zeros(Thunderbolt.num_states(model), length(τ))
us = zeros(Thunderbolt.num_states(model), length(τ))
for (i,t) ∈ enumerate(τ)
    calcium = Ca(t)
    sarcomere_length = Sl(t)
    sarcomere_velocity = (Sl(t+1e-8) - sarcomere_length)/1e-8 # TODO via AD
    Thunderbolt.rhs!(du, u, Vec((0.0,)), 0.0, Thunderbolt.RDQ20MFModel(;
        calcium_field = ConstantCoefficient(calcium),
        sarcomere_length = ConstantCoefficient(sarcomere_length),
        sarcomere_velocity = ConstantCoefficient(sarcomere_velocity)
    ))
    u .+= dt*du
    us[:,i] .= u
end
