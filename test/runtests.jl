using Thunderbolt
using ParallelTestRunner

# Each test file runs in its own module, on a worker process it shares with the other files that
# worker picks up. That is why every file under `test/` carries its own `using` header and includes
# `testfixtures.jl` itself — keep it that way, or it will pass here and fail when run on its own (and
# vice versa).
#
# Useful invocations:
#   julia --project=. -e 'using Pkg; Pkg.test()'                         # all files, parallel
#   julia --project=. -e 'using Pkg; Pkg.test(test_args=["--jobs=1"])'   # serial, for debugging
#   julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_mesh"])'  # one file (prefix match)
#   julia --project=. -e 'using Pkg; Pkg.test(test_args=["--list"])'     # show what would run

const TESTDIR = @__DIR__

testsuite = find_tests(TESTDIR)   # NOT the `pwd()` default: from the repo root that walks docs/, bak/, …

# `find_tests` picks up *every* `.jl` file under `test/`, so what the suite is composed of is decided
# here and not by the directory layout.
#
# Not part of the suite:
#   testfixtures — shared helpers, included by the files that need them
#   gpu/*        — needs CUDA and its own project, no CI job yet
#   data/*       — fixture data (meshes, exported fields), read by the tests that need it
#
# Part of the suite, less obviously so:
#   validation/* — `land2015` reproduces a published benchmark; slow, but it runs on every invocation
#   integration/* — see the worker configuration below
delete!(testsuite, "testfixtures")
for name in collect(keys(testsuite))
    (startswith(name, "gpu/") || startswith(name, "data/")) && delete!(testsuite, name)
end

# `default_njobs()` maximises workers against CPU count and free memory, but it does not know that the
# integration workers below each take `INTEGRATION_THREADS` threads for the per-color assembly. The
# useful bound is therefore cores ÷ threads-per-worker, not cores. Measured on a 16-core box:
# 4 jobs = 190 s, the 11-job default = 340 s, serial = 520 s. Explicit `--jobs=N` still wins.
const INTEGRATION_THREADS = max(1, min(4, Sys.CPU_THREADS ÷ 4))
default_jobs() = clamp(Sys.CPU_THREADS ÷ INTEGRATION_THREADS, 1, 4)

argv = copy(ARGS)
any(startswith("--jobs"), argv) || push!(argv, "--jobs=$(default_jobs())")

args = parse_args(argv)
filter_tests!(testsuite, args)

# Loaded once per test file, into the fresh module that file gets.
const init_code = quote
    using Test
    using Thunderbolt
end

# Every worker gets `INTEGRATION_THREADS` threads. `addworker` otherwise pins JULIA_NUM_THREADS=1,
# which would silently drop the coverage of the threaded per-color assembly the integration tests
# rely on -- `test/integration/test_fsi.jl`'s threaded-vs-sequential equivalence testset degenerates
# to a no-op on a single-threaded worker; `-t` overrides the env var. Passing `exeflags` here rather
# than through a per-file `test_worker` keeps the process count at `jobs`: a `test_worker` spawns its
# own worker while the file still holds a pool slot.
# Keep the product of jobs x threads at or below the core count — Polyester spins, so
# oversubscription hurts more than it helps.
runtests(Thunderbolt, args; testsuite, init_code, exeflags = ["--threads=$(INTEGRATION_THREADS)"])
