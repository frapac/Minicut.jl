#=
    Compares Normal-SDDP (van Ackooij et al. 2016) and Regularized SDDP
=#

using Minicut
using Gurobi

using DataFrames, CSV, JLD2

include("brazilian.jl")
if !@isdefined GRB_ENV
    const GRB_ENV = Gurobi.Env(output_flag = 0)
end

# Intermediate size Brazilian problem
T = 12
nb_scenarios = 20
bhm = BrazilianHydroModel(; T=T, nscen=nb_scenarios) 

# Initialization
lower_bound = -1e8
max_iter = 200
nx = Minicut.number_states(bhm)
optimizer_lp = () -> Gurobi.Optimizer(GRB_ENV)
optimizer_qp = () -> Gurobi.Optimizer(GRB_ENV)

valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
verbose = 1
x0 = bhm.x0

# # Solve with SDDP
# res_sddp = Minicut.sddp(bhm, x0, optimizer_lp; n_iter=max_iter, verbose= verbose, lower_bound = lower_bound, saving_data = true, valid_statuses = valid_statuses);

# Solve with Dual SDDP
# res_dualsddp = Minicut.dualsddp(bhm, x0, optimizer_lp ; n_iter=max_iter, verbose= verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, saving_data = true);

# Solve with Regularized QP SDDP given V and D
# V0 = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
# D0 = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
V300 = load("examples/brazilian/T12_N20/primal/V_1000.jld2", "V")
D1000 = load("examples/brazilian/T12_N20/dual/D_1000.jld2", "D")
res_regQP = Minicut.regularizedsddp_givenVD(bhm, x0, optimizer_lp, optimizer_qp, V300, D1000; τ=5e10, n_iter=max_iter, verbose = verbose,  valid_statuses = valid_statuses, mode = 2, saving_data = false); 

println("--"^10)

# # Solve with Normal SDDP
# n_forward = 10
# max_iter_normal = max_iter ÷ n_forward
# verbose_normal = verbose ÷ n_forward
# # itermax = max_iter
# res_normal = Minicut.normalsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=max_iter_normal, n_forward = n_forward, verbose=verbose_normal, upper_bound = upper_bound);

# # Solve with Normal Solution QP SDDP
# res_normsolQP = Minicut.normalsolutionsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=max_iter, verbose = verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, mode = 1); 

# # Solve with Normal Solution LP SDDP
# res_normsol = Minicut.normalsolutionsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=max_iter, verbose = verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, mode = 3); 

# res_normsol = Minicut.normalsolutionsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e10, n_iter=max_iter, verbose = verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, mode = 3); 
