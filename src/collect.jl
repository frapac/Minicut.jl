# To merge into utils? (take care of types that need to be defined before calling some functions here)

# One DataFrame that holds the problem data (horizon, name of pb, n_scenarios per stage...) and one that holds timers of the algorithm run (time forward steps, backward steps...) 
function init_data(
    solver::AbstractSDDP,
    models::Array{JuMP.Model},
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array,
    allowed_time = 300,
    n_cycle = 20,
    n_iter=100,
    n_pruning = 100,
    n_warmup = 50,
    )
    T = length(V)
    # Saving problem data
    pb_data = DataFrame()
    pb_data[!, :horizon] = [T]
    pb_data[!, :n_x] = [length(x₀)]
    pb_data[!, :n_u] = [length(models[1][_CURRENT_CONTROL])] # Assuming it's constant in t (same for n_x)
    pb_data[!, :allowed_time] = [allowed_time]
    pb_data[!, :n_cycle] = [n_cycle]
    pb_data[!, :n_iter] = [n_iter]
    pb_data[!, :n_pruning] = [n_pruning] 
    pb_data[!, :n_warmup] = [n_warmup]
    pb_data[!, :time_warmup] = [0.0]
    

    # Initializing the DataFrames to be filled during the run
    df_timers = DataFrame()
    df_timers[!, :iteration] = 1:n_iter
    df_timers[!, :time_iter] .= 0.0
    df_timers[!, :time_primal_forward] .= 0.0
    df_timers[!, :time_primal_backward] .= 0.0
    df_timers[!, :time_dual_forward] .= 0.0
    df_timers[!, :time_dual_backward] .= 0.0
    df_timers[!, :time_pruning] .= 0.0

    df_ub = DataFrame(Inf*ones(Float64, (n_iter, T)), :auto)
    insertcols!(df_ub, 1, :iteration => 1:n_iter)

    df_lb = DataFrame(-Inf*ones(Float64, (n_iter, T)), :auto)
    insertcols!(df_lb, 1, :iteration => 1:n_iter)

    return pb_data, df_timers, df_ub, df_lb
end

function init_data(
    solver::NormalSDDP,
    models::Array{JuMP.Model},
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array,
    allowed_time = 300,
    n_batch = 10,
    n_cycle = 20,
    n_iter=100,
    n_pruning = 100,
    n_warmup = 50,
    )
    T = length(V)
    # Saving problem data
    pb_data = DataFrame()
    pb_data[!, :horizon] = [T]
    pb_data[!, :n_x] = [length(x₀)]
    pb_data[!, :n_u] = [length(models[1][_CURRENT_CONTROL])] # Assuming it's constant in t (same for n_x)
    pb_data[!, :allowed_time] = [allowed_time]
    pb_data[!, :n_batch] = [n_batch]
    pb_data[!, :n_cycle] = [n_cycle]
    pb_data[!, :n_iter] = [n_iter]
    pb_data[!, :n_pruning] = [n_pruning] 
    pb_data[!, :n_warmup] = [n_warmup]
    pb_data[!, :time_warmup] = [0.0]
    
    # Initializing the DataFrames to be filled during the run
    df_timers = DataFrame()
    df_timers[!, :iteration] = 1:n_iter
    df_timers[!, :time_iter] .= 0.0
    df_timers[!, :time_primal_forward] .= 0.0
    df_timers[!, :time_primal_backward] .= 0.0
    df_timers[!, :time_pruning] .= 0.0

    df_lb = DataFrame(-Inf*ones(Float64, (n_iter, T)), :auto)
    insertcols!(df_lb, 1, :iteration => 1:n_iter)

    return pb_data, df_timers, df_lb
end

function init_data(
    solver::SDDP,
    models::Array{JuMP.Model},
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array,
    allowed_time = 300,
    n_iter=100,
    )
    T = length(V)
    # Saving problem data
    pb_data = DataFrame()
    pb_data[!, :horizon] = [T]
    pb_data[!, :n_x] = [length(x₀)]
    pb_data[!, :n_u] = [length(models[1][_CURRENT_CONTROL])] # Assuming it's constant in t (same for n_x)
    pb_data[!, :allowed_time] = [allowed_time]
    pb_data[!, :n_iter] = [n_iter]
    
    # Initializing the DataFrames to be filled during the run
    df_timers = DataFrame()
    df_timers[!, :iteration] = 1:n_iter
    df_timers[!, :time_iter] .= 0.0
    df_timers[!, :time_primal_forward] .= 0.0
    df_timers[!, :time_primal_backward] .= 0.0
    df_timers[!, :time_pruning] .= 0.0

    df_lb = DataFrame(-Inf*ones(Float64, (n_iter, T)), :auto)
    insertcols!(df_lb, 1, :iteration => 1:n_iter)

    return pb_data, df_timers, df_lb
end