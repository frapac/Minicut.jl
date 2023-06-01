include("../examples/brazilian/brazilian.jl")

using Gurobi, CSV, DataFrames
const GRB_ENV = Gurobi.Env(output_flag = 0)

@testset "Creating DataFrames" begin 
    T = 5
    lower_bound = -1e9
    max_iter = 1000
    n_scenarios = 10
    n_cycle = 10
    n_pruning = 100
    allowed_time = 120
    n_iter = 500
    n_simus = 1000 # for initial ub by MonteCarlo
    n_warmup = 0
    n_simus = 1000
    verbose = 0
    valid_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    lower_bound = -1e9

    bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
    nx = Minicut.number_states(bhm)
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]

    
    optimizer = () -> Gurobi.Optimizer(GRB_ENV)
    sddp = Minicut.SDDP(optimizer, valid_statuses)
    models = Minicut.build_stage_models(sddp, bhm, V)
    
    pb_data, df_timers, df_ub, df_lb, df_traj  = Minicut.init_data(
        sddp,
        models,
        bhm,
        V,
        bhm.x0,
        allowed_time,
        n_cycle,
        n_iter,
        n_pruning,
        n_warmup,
        )

    @test pbdata[!, :horizon][1] == T
    @test pbdata[!, :n_u][1] == 144
end