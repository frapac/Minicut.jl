
@testset "Minicut interface" begin
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    max_sddp_iter = 100

    ndigits = 6
    # Horizon
    T = 4
    # Initial position
    x0 = [8.0]
    # Build model
    wdm = WaterDamModel(T; nbins=3)

    # Extensive form
    sol_extensive = Minicut.extensive(wdm, x0, optimizer)

    # Primal SDDP
    sol_primal = Minicut.sddp(wdm, x0, optimizer; n_iter=max_sddp_iter, verbose=0)
    # Dual SDDP
    sol_dual = Minicut.dualsddp(wdm, x0, optimizer; n_iter=max_sddp_iter, verbose=0)
    # Mixed SDDP
    sol_mixed = Minicut.mixedsddp(wdm, x0, optimizer; n_iter=max_sddp_iter, verbose=0)

    lb = trunc(sol_primal.lower_bound, digits=ndigits)
    ub = trunc(sol_dual.upper_bound, digits=ndigits)
    opt = trunc(sol_extensive.optimum, digits=ndigits)
    @test_broken lb <= opt <= ub

    lb_mixed = trunc(sol_mixed.lower_bound, digits=ndigits)
    ub_mixed = trunc(sol_mixed.upper_bound, digits=ndigits)
    @test lb_mixed <= ub_mixed
end

