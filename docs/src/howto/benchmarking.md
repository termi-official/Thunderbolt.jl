# Benchmarking

To investiage the performance we can use the following code snippet, which should be self-explanatory

```julia
using Thunderbolt.TimerOutputs
TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
run_simulation()
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)
```

It makes sense to make sure the code is properly precompiled before benchmarkins, e.g. by calling `run_simulation()` once before running the code snippet.

Internally we use [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) for code annotations,
marking performance critical sections.

More guides coming soon...
