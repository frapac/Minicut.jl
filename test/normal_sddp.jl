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
    lower_bound = -3.0e8
    lip_ub = 1e4
    lip_lb = -1e4  

    max_iter = 150
    n_cycle = 1
    n_pruning = 1
    allowed_time = 600
    n_warmup = 0
    verbose = 0

    bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
    nx = Minicut.number_states(bhm)
    x0 = bhm.x0   

    # # Solve with SDDP
    # optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    # solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    # V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T] # V will contain sddp iters until saturation
    # models = Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=50, allowed_time = allowed_time, saving_data=true)
    # objective_sddp = V[1](x0)

    # for t in 1:T
    #     println("L_$t max = $(Minicut.lipschitz_constant(V[t], 2))")
    # end

    # # Solve with Mixed Primal Dual SDDP
    # mixed_sol = Minicut.mixedsddp(bhm, x0, optimizer; seed=0, n_iter= max_iter*10, verbose = 20, lower_bound=lower_bound, lip_ub=+1e10, lip_lb=-1e10, valid_statuses=[MOI.OPTIMAL], allowed_time = 1200, saving_data = true)


    # # Solve with Dual SDDP
    # optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    # dual_sddp = Minicut.DualSDDP(optimizer; lip_lb=-1e6, lip_ub=1e6)
    # D = [Minicut.PolyhedralFunction(nx, lower_bound) for t in 1:T]
    # dual_models = Minicut.solve!(dual_sddp, bhm, D, x0; n_iter=max_iter*10, verbose=10, saving_data = true, allowed_time = allowed_time)

    # # Solve with normal SDDP
    # optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    # solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    # V1 = deepcopy(V)
    # normal_sol = Minicut.normalsddp(bhm, x0, optimizer, V1; n_iter=max_iter*10, verbose=10, τ=1e8, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED], saving_data = true)
    # objective_normal = normal_sol.lower_bound
    

    # Solve with regularized SDDP 2
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    # V3 = deepcopy(V)
    V3 = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    reg_sol2 = Minicut.regularizedsddp2(bhm, x0, optimizer, V3, D; n_iter=3*max_iter, verbose=10, τ=1e8, lower_bound=lower_bound, n_cycle=n_cycle, n_pruning = n_pruning, allowed_time = allowed_time, n_warmup = n_warmup, valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED], saving_data = true, lip_ub = lip_ub, lip_lb = lip_lb)
    objective_primal2 = reg_sol2.lower_bound
    objective_dual2 = reg_sol2.upper_bound

    n_points = 1000
    box = x0 .+ repeat(range(0, 10000, length=n_points)', nx, 1)
    val_diff = zeros(Float64, n_points, T) 
    for t in 1:T
        for i in 1:n_points
            val_diff[i,t] = Minicut.difference(optimizer, box[:,i], V3[t], D[t])
        end
    end
    println("Number of points with negative gap = $(length(filter(x-> x<0, val_diff)))")


    # println("Regularized SDDP gap..............: $((abs(objective_dual2 - objective_primal2) / abs(objective_dual2))*100)% ")

    # # Keep doing SDDP 
    # Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=50, allowed_time = allowed_time, saving_data=true)

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

