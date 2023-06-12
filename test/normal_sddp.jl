using JuMP
using HiGHS
using Statistics
using Gurobi
using Random
using Plots

@testset "Normal SDDP" begin 
    Random.seed!(2713)
    n_scenarios = 10
    T = 5
    lower_bound = -1e9
    max_iter = 1000
    n_simus = 100
    n_cycle = 10
    n_pruning = 100
    allowed_time = 60
    n_simus = 1000 # for initial ub by MonteCarlo
    n_warmup = 0
    n_simus = 1000
    verbose = 0

    bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
    nx = Minicut.number_states(bhm)
    x0 = bhm.x0   

    # Solve with SDDP 
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T] # V will contain sddp iters until saturation
    models = Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=verbose)
    objective_sddp = V[1](x0)
    println("SDDP value........................: $(objective_sddp)")
    # Statistical upper bound
    scenarios = Minicut.sample(Minicut.uncertainties(bhm), n_simus)
    costs = Minicut.simulate!(solver, bhm, models, x0, scenarios)
    stat_ub = mean(costs) + 1.96 * std(costs) / sqrt(n_simus)
    stat_relgap = (abs(stat_ub - objective_sddp) / abs(stat_ub))*100
    println("SDDP statistical gap..............: $(stat_relgap)% ")

    # Solve with normal SDDP
    V1 = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
    norm_sol = Minicut.normalsddp(bhm, x0, optimizer, V1; n_iter=max_iter, verbose=verbose, τ=1e8, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
    objective_normal = normal_sol.lower_bound
    println("SDDP value........................: $(objective_normal)")

    # Statistical upper bound
    scenarios_normal = Minicut.sample(Minicut.uncertainties(bhm), n_simus)
    costs_normal = Minicut.simulate!(solver, bhm, models, x0, scenarios_normal)
    normal_ub = mean(costs_normal) + 1.96 * std(costs_normal) / sqrt(n_simus)
    stat_normal = (abs(normal_ub - objective_normal) / abs(normal_ub))*100
    println("Normal SDDP statistical gap........: $(stat_normal)% ")
    
end