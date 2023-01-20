using JuMP
using HiGHS
using Statistics

@testset "SDDP: WaterDamModel" begin
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    lower_bound = -1e4
    x0 = [8.0]
    T = 3
    nx = 1
    wdm = WaterDamModel(T)
    Ξ = Minicut.uncertainties(wdm)

    @test Minicut.horizon(wdm) == T

    V = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]

    sddp = Minicut.SDDP(optimizer)

    #=
        First, we test the mechanics of the forward and backward passes.
    =#
    @testset "Forward/backward pass" begin
        # JuMP Model
        models = JuMP.Model[Minicut.stage_model(wdm, t) for t in 1:T]
        for (t, model) in enumerate(models)
            if t < T
                Minicut.initialize!(sddp, model, V[t+1])
            end
            JuMP.set_optimizer(model, optimizer)
        end

        scenario = Minicut.sample(Ξ)

        primal_scenarios = Minicut.forward_pass(sddp, wdm, models, scenario, x0)

        @test isa(primal_scenarios, Array{Float64, 2})
        @test size(primal_scenarios) == (nx, T+1)
        # Test JuMP model is well defined
        @test JuMP.value.(models[1][:xₜ]) == x0
        @test JuMP.value.(models[1][:ξₜ₊₁]) == scenario[:, 1]

        dual_scenarios = Minicut.backward_pass!(sddp, wdm, models, primal_scenarios, V)
        @test size(dual_scenarios) == size(primal_scenarios)
        # Test we have the right amount of cuts.
        @test Minicut.ncuts(V[1]) == 2
    end

    #=
        Test SDDP itself.
    =#
    @testset "SDDP algorithm" begin
        models = Minicut.solve!(sddp, wdm, V, x0; n_iter=10, verbose=0)

        n_scenarios = 1000
        scenarios = Minicut.sample(Ξ, n_scenarios)
        costs = Minicut.simulate!(sddp, wdm, models, x0, scenarios)

        lb = V[1](x0)
        ub = mean(costs) + 1.96 * std(costs) / sqrt(n_scenarios)
        # Test convergence of SDDP
        # TODO: check we satisfy this bound for any random seed
        @test abs(ub - lb) / abs(ub) <= 0.02
    end

    # Test lower bound matches exactly upper bound in deterministic case
    @testset "Deterministic problem" begin
        T = 10
        Vdet = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]

        wdm_determistic = WaterDamModel(T; nbins=1)
        solver = Minicut.SDDP(optimizer)
        models = Minicut.solve!(solver, wdm_determistic, Vdet, x0; n_iter=100, verbose=0)
        #
        n_scenarios = 1
        scenarios = Minicut.sample(Minicut.uncertainties(wdm_determistic), n_scenarios)
        costs = Minicut.simulate!(solver, wdm_determistic, models, x0, scenarios)
        @test mean(costs) ≈ Vdet[1](x0)
    end
end

@testset "Test SDDP against Extensive formulation" begin
    # Build small model
    T, nbins = 3, 2
    wdm = WaterDamModel(T; nbins=nbins)
    nx = Minicut.number_states(wdm)
    x0 = [8.0]

    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    # Solve with SDDP
    lower_bound = -1e4
    V = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]
    solver = Minicut.SDDP(optimizer)
    models = Minicut.solve!(solver, wdm, V, x0; n_iter=10, verbose=0)
    objective_sddp = V[1](x0)

    # Solve with extensive form
    extensive_solver = Minicut.ExtensiveFormulationSolver(optimizer)
    model = Minicut.solve!(extensive_solver, wdm, x0)
    objective_extensive_form = JuMP.objective_value(model)

    @test objective_sddp ≈ objective_extensive_form
end

