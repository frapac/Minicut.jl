using JuMP
using HiGHS
using Statistics
using Gurobi
using Random
using Plots

@testset "SDDP then regularization" begin 
    Random.seed!(2713)
    n_scenarios = 5
    T = 5
    lower_bound = -3.0e6
    max_iter = 150
    n_cycle = 1
    n_pruning = 1
    allowed_time = 600
    n_warmup = 0
    verbose = 0

    bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
    nx = Minicut.number_states(bhm)
    x0 = bhm.x0   

    # 1000 SDDP step
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T] # V will contain sddp iters until saturation
    models = Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=50, allowed_time = allowed_time, saving_data=true)
    objective_sddp = V[1](x0)

    # Solve with normal SDDP
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    V1 = deepcopy(V)
    normal_sol = Minicut.normalsddp(bhm, x0, optimizer, V1; n_iter=max_iter*10, verbose=10, τ=1e8, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED], saving_data = true)
    objective_normal = normal_sol.lower_bound
    

    # Solve with regularized SDDP 2
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    V3 = deepcopy(V)
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    reg_sol2 = Minicut.regularizedsddp2(bhm, x0, optimizer, V3, D; n_iter=10*max_iter, verbose=10, τ=1e8, lower_bound=lower_bound, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED], saving_data = true)
    objective_primal2 = reg_sol2.lower_bound
    objective_dual2 = reg_sol2.upper_bound
    println("Regularized SDDP gap..............: $((abs(objective_dual2 - objective_primal2) / abs(objective_dual2))*100)% ")

    # Keep doing SDDP 
    Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=50, allowed_time = allowed_time, saving_data=true)

    #@test abs(objective_normal - objective_sddp)/abs(objective_sddp) < 0.001
end

# @testset "Vanilla SDDP vs Normal SDDP vs Regularized SDDP" begin 
#     Random.seed!(2713)
#     n_scenarios = 20
#     T = 50
#     lower_bound = -1e9
#     max_iter = 10000
#     n_cycle = 10
#     n_pruning = 100
#     allowed_time = 1200
#     n_warmup = 0
#     verbose = 0

#     bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
#     nx = Minicut.number_states(bhm)
#     x0 = bhm.x0   

#     # # Solve with SDDP 
#     # optimizer = () -> Gurobi.Optimizer(GRB_ENV)
#     # solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
#     # V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T] # V will contain sddp iters until saturation
#     # models = Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=verbose, allowed_time = allowed_time, saving_data=true)
#     # objective_sddp = V[1](x0)

#     # # Solve with normal SDDP
#     # optimizer = () -> Gurobi.Optimizer(GRB_ENV)
#     # solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
#     # V1 = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
#     # normal_sol = Minicut.normalsddp(bhm, x0, optimizer, V1; n_iter=max_iter, verbose=10, τ=1e8, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED], saving_data = false)
#     # objective_normal = normal_sol.lower_bound

#      # Solve with regularized SDDP 2
#     optimizer = () -> Gurobi.Optimizer(GRB_ENV)
#     solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
#     V3 = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
#     D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
#     reg_sol2 = Minicut.regularizedsddp2(bhm, x0, optimizer, V3, D; n_iter=max_iter, verbose=10, τ=1e8, lower_bound=lower_bound, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
#     objective_primal2 = reg_sol2.lower_bound
#     objective_dual2 = reg_sol2.upper_bound
#     println("Regularized SDDP gap..............: $((abs(objective_dual2 - objective_primal2) / abs(objective_dual2))*100)% ")
   
#     @test abs(objective_normal - objective_sddp)/abs(objective_sddp) < 0.001
# end

