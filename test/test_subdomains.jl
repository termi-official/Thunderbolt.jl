@testset "Subdomains" begin
    subdomain_error = ArgumentError("Using DofHandler with multiple subdomains is not currently supported")

    # Dummy values, dof and constraint handler
    grid = generate_grid(Hexahedron, (2,1,1))
    ip = getinterpolation(LagrangeCollection{1}(), RefHexahedron)
    dh = DofHandler(grid)
    sdh_A = SubDofHandler(dh, Set(1))
    Ferrite.add!(sdh_A, :u, ip)
    sdh_B = SubDofHandler(dh, Set(2))
    Ferrite.add!(sdh_B, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    qr = QuadratureRule{RefHexahedron}(2)
    cv = CellValues(qr, ip)

    # Dummy QuasiStaticModel
    struct DummyCalciumHatField end
    microstructure_model = ConstantCoefficient((
    Vec((1.0, 0.0, 0.0)),
    Vec((0.0, 1.0, 0.0)),
    Vec((0.0, 0.0, 1.0)),
    ))
    qsm = GeneralizedHillModel(
        LinYinPassiveModel(),
        ActiveMaterialAdapter(LinYinActiveModel()),
        GMKIncompressibleActiveDeformationGradientModel(),
        PelceSunLangeveld1995Model(;calcium_field=DummyCalciumHatField()),
        microstructure_model
    )
    # Dummy Coefficients
    analytical_coeff = AnalyticalCoefficient((x,t) -> norm(x)+t, CoordinateSystemCoefficient(CartesianCoordinateSystem(ip^3)))
    spectral_coeff = SpectralTensorCoefficient(
        ConstantCoefficient(SVector((Vec((1.0,0.0)),))),
        ConstantCoefficient(SVector((-1.0,))),
    )
    # Dummy Cache
    element_cache = Thunderbolt.StructuralElementCache(
        qsm,
        Thunderbolt.setup_contraction_model_cache(cv, qsm.contraction_model),
        cv
    )
    # Dummy protocol
    protocol = AnalyticalTransmembraneStimulationProtocol(analytical_coeff, [SVector((0., 0.))])
    # Dummy writers
    io = ParaViewWriter("")
    ioJLD2 = JLD2Writer("") 
    
    @test_throws subdomain_error LVCoordinateSystem(dh, [0.], [0.])
    @test_throws subdomain_error Thunderbolt.compute_chamber_volume(dh, [0.], "top", Thunderbolt.Hirschvogel2016SurrogateVolume)
    @test_throws subdomain_error Thunderbolt.TransientHeatProblem(Thunderbolt.ConductivityToDiffusivityCoefficient(0., 0., 0.), protocol, dh)
    @test_throws subdomain_error Thunderbolt.QuasiStaticNonlinearProblem(dh, ch, qsm, [])
    @test_throws subdomain_error Thunderbolt.AssembledNonlinearOperator([0. 0.;0. 0.], element_cache, (), dh)
    @test_throws subdomain_error Thunderbolt.AssembledBilinearOperator([0. 0.;0. 0.], element_cache, dh)
    @test_throws subdomain_error Thunderbolt.LinearOperator([0.], element_cache, dh)
    @test_throws subdomain_error Thunderbolt.create_linear_operator(dh, NoStimulationProtocol()) 
    @test_throws subdomain_error Thunderbolt.create_linear_operator(dh, protocol)
    @test_throws subdomain_error store_timestep_field!(io, 0, dh, [0.], :u)
    @test_throws subdomain_error store_timestep_field!(io, 0, dh, [0.], "u")
    @test_throws subdomain_error store_coefficient!(io, dh, analytical_coeff, "", 0, QuadratureRuleCollection(1))
    @test_throws subdomain_error store_coefficient!(io, dh, spectral_coeff, "", 0)
    @test_throws subdomain_error store_green_lagrange!(io, dh, [0.], analytical_coeff, spectral_coeff, cv, "", 0)
    @test_throws subdomain_error store_timestep_field!(ioJLD2, 0, dh, [0.], "u")
    @test_throws subdomain_error store_timestep_field!(ioJLD2, 0, dh, [0.], "u")

end
