#=
    Compares Normal-SDDP (van Ackooij et al. 2016) and Regularized SDDP
=#

using Minicut
using Gurobi

include("brazilian.jl")
if !@isdefined GRB_ENV
    const GRB_ENV = Gurobi.Env(output_flag = 0)
end

# Intermediate size Brazilian problem
T = 25
nb_scenarios = 20 
bhm = BrazilianHydroModel(; T=T, nscen=nb_scenarios) 

# Initialization
lower_bound = -1e8
upper_bound = 1e10
max_iter = 250
nx = Minicut.number_states(bhm)
optimizer_lp = () -> Gurobi.Optimizer(GRB_ENV)
optimizer_qp = () -> Gurobi.Optimizer(GRB_ENV)
valid_statuses = [MOI.OPTIMAL]
verbose = 1
x0 = bhm.x0


# Solve with Regularized SDDP 
res_reg = Minicut.regularizedsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=max_iter, verbose = verbose, lower_bound = lower_bound, valid_statuses = valid_statuses); 
# takes 2950 seconds for 500 iterations


# Solve with Normal SDDP
n_forward = 10
#itermax = max_iter ÷ n_forward
itermax = max_iter
res_normal = Minicut.normalsddp(bhm, x0, optimizer_lp, optimizer_qp; τ=1e8, n_iter=itermax, n_forward = n_forward, verbose=verbose, upper_bound = upper_bound);

println("----")

# Solve with SDDP
 res_sddp = Minicut.sddp(bhm, x0, optimizer_lp; n_iter=max_iter, verbose= verbose, lower_bound = lower_bound);

