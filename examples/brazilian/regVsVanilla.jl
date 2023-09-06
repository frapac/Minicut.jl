#=
    Compares Normal-SDDP (van Ackooij et al. 2016) and Regularized SDDP
=#

using Minicut
using Gurobi

using DataFrames, CSV

include("brazilian.jl")
if !@isdefined GRB_ENV
    const GRB_ENV = Gurobi.Env(output_flag = 0)
end

# Intermediate size Brazilian problem
T = 5
nb_scenarios = 5
bhm = BrazilianHydroModel(; T=T, nscen=nb_scenarios) 

# Initialization
lower_bound = -1e10
upper_bound = 1e10
max_iter = 100
nx = Minicut.number_states(bhm)
optimizer_lp = () -> Gurobi.Optimizer(GRB_ENV)
optimizer_qp = () -> Gurobi.Optimizer(GRB_ENV)

valid_statuses = [MOI.OPTIMAL]
verbose = 1
x0 = bhm.x0

# # Solve with Regularized LP SDDP
# res_regLP = Minicut.regularizedsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=max_iter, verbose = verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, mode = 3); 

# Solve with Regularized QP SDDP 
res_regQP = Minicut.regularizedsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=max_iter, verbose = verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, mode = 1, saving_data = true); 

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

# Solve with SDDP
 res_sddp = Minicut.sddp(bhm, x0, optimizer_lp; n_iter=max_iter, verbose= verbose, lower_bound = lower_bound, saving_data = false, valid_statuses = valid_statuses);

# Solve with Dual SDDP
res_dualsddp = Minicut.dualsddp(bhm, x0, optimizer_lp ; n_iter=max_iter, verbose= verbose, lower_bound = lower_bound, valid_statuses = valid_statuses, saving_data = false)

println("--"^10)
