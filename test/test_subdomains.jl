@testset "Subdomains" begin
    subdomain_error = ArgumentError("Using DofHandler with multiple subdomains is not currently supported")

    # Dummy values, dof and constraint handler
    grid = generate_grid(Hexahedron, (2,1,1))
    ipc = LagrangeCollection{1}()
    ip = getinterpolation(ipc, RefHexahedron)
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
    analytical_coeff = AnalyticalCoefficient((x,t) -> norm(x)+t, CoordinateSystemCoefficient(CartesianCoordinateSystem(grid)))
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

    qr = QuadratureRuleCollection(1)
    solver = BackwardEulerSolver()
    
    @test_throws subdomain_error LVCoordinateSystem(dh, ipc, [0.], [0.], [0.])
    @test_throws subdomain_error TransientHeatFunction(Thunderbolt.ConductivityToDiffusivityCoefficient(0., 0., 0.), protocol, dh)
    @test_throws subdomain_error QuasiStaticNonlinearFunction(dh, ch, qsm, [])
    @test_throws subdomain_error Thunderbolt.AssembledNonlinearOperator([0. 0.;0. 0.], element_cache, (), (), dh)
    @test_throws subdomain_error Thunderbolt.AssembledBilinearOperator([0. 0.;0. 0.], element_cache, dh)
    @test_throws subdomain_error Thunderbolt.LinearOperator([0.], element_cache, dh)
    @test_throws subdomain_error Thunderbolt.setup_operator(NoStimulationProtocol(), solver, dh, :u, qr)
    @test_throws subdomain_error Thunderbolt.setup_operator(protocol, solver, dh, :u, qr)
    @test_throws subdomain_error store_coefficient!(io, dh, analytical_coeff, "", 0, QuadratureRuleCollection(1))
    @test_throws subdomain_error store_coefficient!(io, dh, spectral_coeff, "", 0)
    @test_throws subdomain_error store_green_lagrange!(io, dh, [0.], analytical_coeff, spectral_coeff, cv, "", 0)
end
