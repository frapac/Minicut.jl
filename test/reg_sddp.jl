using JuMP
using HiGHS
using Statistics
using Gurobi
using Random
const GRB_ENV = Gurobi.Env(output_flag = 0)

# @testset "Regularized SDDP: WaterDamModel" begin
#     optimizer = JuMP.optimizer_with_attributes(
#         HiGHS.Optimizer, "output_flag" => false,
#     )
#     lower_bound = -1e4
#     x0 = [8.0]
#     T = 3
#     nx = 1
#     wdm = WaterDamModel(T)
#     Ξ = Minicut.uncertainties(wdm)

#     @test Minicut.horizon(wdm) == T

#     V = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]

#     sddp = Minicut.SDDP(optimizer)

#     #=
#         First, we test the mechanics of the forward and backward passes.
#     =#
#     @testset "Forward/backward pass" begin
#         # JuMP Model
#         models = JuMP.Model[Minicut.stage_model(wdm, t) for t in 1:T]
#         for (t, model) in enumerate(models)
#             if t < T
#                 Minicut.initialize!(sddp, model, V[t+1])
#             end
#             JuMP.set_optimizer(model, optimizer)
#         end

#         scenario = Minicut.sample(Ξ)

#         primal_scenarios = Minicut.forward_pass(sddp, wdm, models, scenario, x0)

#         @test isa(primal_scenarios, Array{Float64,2})
#         @test size(primal_scenarios) == (nx, T + 1)
#         # Test JuMP model is well defined
#         @test JuMP.value.(models[1][Minicut._PREVIOUS_STATE]) == x0
#         @test JuMP.value.(models[1][Minicut._UNCERTAINTIES]) == scenario[:, 1]

#         dual_scenarios = Minicut.backward_pass!(sddp, wdm, models, primal_scenarios, V)
#         @test size(dual_scenarios) == size(primal_scenarios)
#         # Test we have the right amount of cuts.
#         @test Minicut.ncuts(V[1]) == 2
#     end

#     #=
#         Test SDDP itself.
#     =#
#     @testset "SDDP algorithm" begin
#         models = Minicut.solve!(sddp, wdm, V, x0; n_iter=10, verbose=0)

#         n_scenarios = 1000
#         scenarios = Minicut.sample(Ξ, n_scenarios)
#         costs = Minicut.simulate!(sddp, wdm, models, x0, scenarios)

#         lb = V[1](x0)
#         ub = mean(costs) + 1.96 * std(costs) / sqrt(n_scenarios)
#         # Test convergence of SDDP
#         # TODO: check we satisfy this bound for any random seed
#         @test abs(ub - lb) / abs(ub) <= 0.02
#     end

#     # Test lower bound matches exactly upper bound in deterministic case
#     @testset "Deterministic problem" begin
#         T = 10
#         Vdet = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]

#         wdm_determistic = WaterDamModel(T; nbins=1)
#         solver = Minicut.SDDP(optimizer)
#         models = Minicut.solve!(solver, wdm_determistic, Vdet, x0; n_iter=100, verbose=0)
#         #
#         n_scenarios = 1
#         scenarios = Minicut.sample(Minicut.uncertainties(wdm_determistic), n_scenarios)
#         costs = Minicut.simulate!(solver, wdm_determistic, models, x0, scenarios)
#         @test mean(costs) ≈ Vdet[1](x0)
#     end
# end

# @testset "Test reg_SDDP vs SDDP" begin
#     # Build small model
#     T, nbins = 5, 3
#     wdm = WaterDamModel(T; nbins=nbins)
#     nx = Minicut.number_states(wdm)
#     x0 = [8.0]

#     optimizer = JuMP.optimizer_with_attributes(
#         HiGHS.Optimizer, "output_flag" => false,
#     )
#     # Solve with SDDP
#     lower_bound = -1e4
#     V = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]
#     solver = Minicut.SDDP(optimizer)
#     models = Minicut.solve!(solver, wdm, V, x0; n_iter=20, verbose=1)
#     objective_sddp = V[1](x0)

#     # Solve with regularized SDDP
#     reg_sol = Minicut.regularizedsddp(wdm, x0, optimizer; n_iter=30, verbose=1, τ=1e10)
#     objective_primal = reg_sol.lower_bound
#     objective_dual = reg_sol.upper_bound
#     println("SDDP : $objective_sddp ; Reg SDDP primal : $objective_primal ; Reg SDDP dual : $objective_dual")
#     @test abs(objective_sddp - objective_primal) / objective_sddp < 0.01
# end

@testset "Test reg SDDP brazilian" begin
    Random.seed!(2713)
    nscenarios = 50
    T = 25
    lower_bound = -1e9
    max_iter = 300
    nsimus = 100
    n_warming = 50
    n_cycle = 10
    n_prunning = 100
    allowed_time = 600

    bhm = BrazilianHydroModel(; T=T, nscen=nscenarios)
    nx = Minicut.number_states(bhm)
    x0 = bhm.x0
    # Initialize value functions.
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]

    # Solve with SDDP
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    models = Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=10)

    # # Simulation
    # scenarios = Minicut.sample(Minicut.uncertainties(bhm), nsimus)
    # costs = Minicut.simulate!(solver, bhm, models, x0, scenarios)
    # stat_ub = mean(costs) + 1.96 * std(costs) / sqrt(nsimus)
    # objective_sddp = V[1](x0)
    # println("Final statistical gap: ", abs(stat_ub - objective_sddp) / abs(stat_ub))

    # # Solve with regularized SDDP
    # reg_sol = Minicut.regularizedsddp(bhm, x0, optimizer; n_iter=max_iter, verbose=10, τ=1e8, lower_bound=lower_bound, n_warming=n_warming)
    # objective_primal = reg_sol.lower_bound
    # objective_dual = reg_sol.upper_bound
    # println("SDDP: $objective_sddp ; Reg SDDP primal: $objective_primal ; Reg SDDP dual: $objective_dual")

    # Solve with regularized SDDP 2
    reg_sol2 = Minicut.regularizedsddp2(bhm, x0, optimizer; n_iter=max_iter, verbose=10, τ=1e8, lower_bound=lower_bound, n_cycle=n_cycle, n_prunning = n_prunning, allowed_time = allowed_time)
    objective_primal2 = reg_sol2.lower_bound
    objective_dual2 = reg_sol2.upper_bound
    
    #println("SDDP: $objective_sddp ; Reg SDDP primal 2: $objective_primal2 ; Reg SDDP dual 2: $objective_dual2 ")

    #@test abs(objective_primal - stat_ub) / stat_ub < 0.01
end

