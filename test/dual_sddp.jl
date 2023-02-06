
@testset "DualSDDP: WaterDamModel" begin
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    niter = 100

    nx = 1
    # Horizon
    T = 4
    # Initial position
    x0 = [8.0]
    # Build model
    wdm = WaterDamModel(T; nbins=3)
    Ξ = Minicut.uncertainties(wdm)

    # Initial cuts.
    lower_bound = -1e8
    V = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]

    #=
        Solve with Primal SDDP
    =#
    primal_sddp = Minicut.SDDP(optimizer)
    primal_models = Minicut.solve!(primal_sddp, wdm, V, x0; n_iter=niter, verbose=0)
    lb = V[1](x0)

    #=
        Solve with extensive formulation
    =#
    extensive_solver = Minicut.ExtensiveFormulationSolver(optimizer)
    extmodel = Minicut.solve!(extensive_solver, wdm, x0)
    ref = JuMP.objective_value(extmodel)

    #=
        Solve with Dual SDDP
    =#
    dual_sddp = Minicut.DualSDDP(optimizer; lip_lb=-1e6, lip_ub=1e6)
    D = [Minicut.PolyhedralFunction(nx) for t in 1:T]
    dual_models = Minicut.solve!(dual_sddp, wdm, D, x0; n_iter=0, verbose=0)

    # Build initial cuts with primal solution
    primtraj = Minicut.forward_pass(primal_sddp, wdm, primal_models, Minicut.sample(Ξ), x0)
    dualtraj = Minicut.backward_pass!(primal_sddp, wdm, primal_models, primtraj, V)
    Minicut.backward_pass!(dual_sddp, wdm, dual_models, dualtraj, D)

    dual_models = Minicut.solve!(dual_sddp, wdm, D, x0; n_iter=niter, verbose=0)
    ub, p0 = Minicut.fenchel_transform(dual_sddp, D[1], x0)

    @test isa(p0, Vector)
    # Test all results are matching
    @test lb <= ub
    @test isapprox(lb, ub, atol=1e-6)
    @test isapprox(lb, ref, atol=1e-6)
end
