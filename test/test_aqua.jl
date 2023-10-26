using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        Thunderbolt;
        ambiguities=false,
        # unbound_args=true,
        # undefined_exports=true,
        # project_extras=true,
        # stale_deps=(ignore=[:SomePackage],),
        # deps_compat=(ignore=[:SomeOtherPackage],),
        # piracy=false,
    )
    Aqua.test_ambiguities(Thunderbolt) # see https://github.com/JuliaTesting/Aqua.jl/issues/77
end

