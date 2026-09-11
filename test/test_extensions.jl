# Package-extension load coverage.
#
# `ThunderboltKernelAbstractionsExt` is triggered by KernelAbstractions ALONE, and
# KernelAbstractions is a hard dependency of CUDA.jl — so every CUDA user loads
# this extension whether or not they touch the device seam. Its own body is
# version-gated on a FerriteOperators surface that the registered 0.4.0 does not
# carry, and a gate that mis-fires (naming the newer seam while it is absent, or
# skipping the block while it is present) is a load-time failure for those users.
#
# None of that needs a GPU: loading KernelAbstractions is the whole trigger. The
# assertions below therefore belong in the ordinary suite, not in `test/gpu/`,
# which this repo's runner excludes.

using Test, Thunderbolt
using KernelAbstractions
import FerriteOperators

@testset "package extensions" begin
    @testset "the KernelAbstractions extension loads" begin
        ext = Base.get_extension(Thunderbolt, :ThunderboltKernelAbstractionsExt)
        @test ext !== nothing

        # The guard's own predicate, and the binding that exists only when the
        # guarded block ran. Asserting the EQUIVALENCE is what makes this test
        # meaningful on both stacks: against the registered FerriteOperators the
        # block must stay inactive, against a checkout carrying the device seam it
        # must run — and a gate naming a seam it did not check would fail the load
        # outright, before this file gets to assert anything.
        @test isdefined(ext, :_grid_backed_handler) ==
            isdefined(FerriteOperators, :device_worker_view)
    end
end
