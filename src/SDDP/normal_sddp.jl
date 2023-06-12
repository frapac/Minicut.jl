
struct NormalSDDP <: AbstractSDDP
    primal_sddp::SDDP
    mixing::Float64
    name::String
end 


function solve!(
    solver::NormalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    allowed_time = 300,
    n_cycle = 20,
    n_batch = 10,
    n_iter=100,
    n_pruning = 100,
    n_warmup = 50,
    τ=1e8,
    verbose::Int=1,
    saving_data= true,
    upper_bound = 1e9
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)

    # Saving run data in a DataFrame pb_data, df_timers, df_ub, df_lb, df_traj
    run_data, run_timers, run_ub, run_lb = init_data(solver.primal_sddp,
    primal_models,
    hdm,
    V,
    x₀,
    allowed_time,
    n_cycle,
    n_iter,
    n_pruning,
    n_warmup,
    )
    # Warmup
    
    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
    end

    if n_warmup > 0
        println("Warming up")
        V = warmup!(solver, primal_models, hdm, V, x₀; verbose=verbose, n_warmup = n_warmup, n_pruning = n_pruning)
    end 

    if verbose > 0
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end
    
    # Timers
    time_mainrun = time()

    # Run
    primal_trajectories = Array{Matrix{Float64}}(undef, n_batch)
    ubs = upper_bound*ones(Float64, horizon(hdm), n_batch) # One upperbound per node of the scenario tree

    for i in 1:n_iter
        tic_iter = time()
        scenarios = sample(Ξ, n_batch)
        # Compute regularized forward passes, update upperbounds at the end of it
        for k in 1:n_batch
            primal_trajectories[k] = normal_forward_pass!(solver, hdm, primal_models, V, scenarios[k], x₀, τ, ubs)
        end
        # Compute backward passes
        backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories, V)
        # Pruning    
        if  j == 1
            tic = time()
            V = pruning(V, primal_trajectories; verbose = verbose)
            run_timers[i, :time_pruning] += time() - tic 
        end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        run_timers[i, :time_iter] += time() - tic_iter
        
        if saving_data
            for t in 1:horizon(hdm)
                run_lb[i,t+1] = V[t](primal_trajectories[j][:, t]) # not lb on opt values, but lb on this traj
            end
        end
        # Check if allowed time is over
        if time() - tic > allowed_time
            break
        end
    end

    status = MOI.ITERATION_LIMIT
    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %10s\n\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf("Max number of iterations.........: %7i\n", n_iter)
        @printf("Main loop wall-clock time (sec)..: %7.3f\n\n", time() - time_mainrun)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    #run_data, run_timers, run_ub, run_lb, run_traj
    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_rundata.csv", run_data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runtimers.csv", run_timers)
        CSV.write(lowercase(split(name(hdm))[1])*"_runub.csv", run_ub) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runlb.csv", run_lb) 
    end 
    return (primal_models, dual_models)
end


function cost_tj!(
    hdm::HazardDecisionModel,
    Ξ::Vector{DiscreteRandomVariable{T}},
    cum_costs::Matrix{T}, # (i,t) matrix of size nb_scenarios x T containing cum cost for scenario i at time t
    scenarios::Array{Matrix{T}},
    t::Int,
    j::Int,
    ubs::Matrix{T}
)  where T
    # Grouping the scenarios with common history that go through (t,j)
    list_scenar, hist = histories_tj(hdm, scenarios, Ξ, t, j)
    for k in 1:length(hist)
        weights = [weight(hdm, list_scenar, Ξ)]
        remaining_costs = [cum_costs[i,T+1] - cum_costs[i, t+1] for i in list_scenar]
        futur_expectation = sum(weights[i]*remaining_costs[i]  for i in 1:length(list_scenar)) /sum(weights)
        for i in list_scenar[k]
            ubs[t,j] = min(ubs[t,j], cum_costs[i,t+1] + futur_expectation)
        end
    end
end

# Checks whether path history already seen, if not, add new path history. Then add the scenario index to associated path history
function histories_tj(
    hdm::HazardDecisionModel,
    scenarios::Array{Matrix{T}},
    Ξ::Vector{DiscreteRandomVariable{T}},
    t::Int,
    j::Int
) where T
    list_scenar = Vector{Vector{Int64}}(undef, 0)
    hist = Vector{Vector{Int64}}(undef, 0)
    for i in 1:length(scenarios)
        path = scenario_path(hdm, scenarios[i], Ξ)
        if path[t] == j
            test = true
            k = 1
            while test
                if k > length(list_scenar) # If this history has not been seen before, add it
                    push!(hist, path[1:t])
                    push!(list_scenar, [i])
                    test = false 
                end 
                ## Otherwise, history has been seen, add it to associated list 
                if (path[1:t] == hist[k]) && ([i] != list_scenar[k])
                    push!(list_scenar[k], i)
                    test = false
                end
                k += 1
            end
        end
    end
    return list_scenar, hist
end

function update_ubs!(
    hdm::HazardDecisionModel,
    Ξ::Vector{DiscreteRandomVariable{T}},
    cum_costs::Matrix{T},
    scenarios::Array{Matrix{T}},
    J::Vector{Int},
    ubs::Matrix{T}
) where T
    for t in 1:T
        cost_tj!(hdm, Ξ, cum_costs, scenarios, t, J[t], ubs)
    end
end

function reg_forward_pass!(
    normal_sddp::NormalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
    τ::Float64,
    ubs::Array{Float64, 2},
    cum_costs::Array{Float64,2}, 
    J::Vector{Int}
)
    Ξ = uncertainties(hdm)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    for (t, ξₜ₊₁) in enumerate(eachcol(uncertainty_scenario))
        ξ = collect(ξₜ₊₁)
        # Lower-bound
        lb = lowerbound(normal_sddp.primal_sddp, primal_models[t], xₜ, ξ)
        # Upper-bound
        j = find_outcome(Ξ[t], ξ)
        J[t] = j
        ub = ubs[t, j]
        #relative_gap = abs((ub - lb)/lb)
        #mixing = min(relative_gap, 1)
        mixing = 0.5/t # Implementation details p.25 [van Ackooij et al. (2019)]
        ℓ = mixing * lb + (1.0 - mixing) * ub
        model = stage_model(hdm, t)
        xₜ,cost = next!(normal_sddp, model, V, xₜ, ξ, ℓ, τ, t, horizon(hdm))
        cum_costs[t+1] = cum_costs[t] + cost
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end

function next!(
    normalsddp::NormalSDDP,
    model::JuMP.Model,
    V::Vector{PolyhedralFunction},
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64},
    ℓ::Float64,
    τ::Float64,
    t::Int,
    T::Int,
)
    JuMP.set_optimizer(model, normalsddp.primal_sddp.optimizer)
    solve_stage_problem!(normalsddp.primal_sddp, model, V, xₜ, ξₜ₊₁, ℓ, τ, t, T)
    if t < T
        return value.(model[_CURRENT_STATE]), objective_value(model) - value(model[_VALUE_FUNCTION])
    else 
        return value.(model[_CURRENT_STATE]), objective_value(model)
    end
end



function normal_forward_pass!(
    normalsddp::NormalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    τ::Float64,
    ubs::Array{Float64, 2}
)
    horizon = size(uncertainty_scenario, 2)
    primal_trajectory = fill(0.0, length(initial_state), horizon + 1)
    cum_costs = fill(0.0, length(initial_state), horizon + 1)
    J = zeros(Int64, horizon)
    primal_trajectory = reg_forward_pass!(normalsddp, hdm, primal_models, V, uncertainty_scenario, initial_state, primal_trajectory, τ, ubs, cum_costs, J)
    # Added step compared to regularized_sddp where the bounds are update during the forward step
    update_ubs!(hdm, uncertainties(hdm), cum_costs, uncertainty_scenario, J, ubs)
end

# Helper function
function normalsddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer,
    V::Array{PolyhedralFunction};
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    valid_statuses=[MOI.OPTIMAL],
    n_cycle= 10,
    n_pruning = 100,
    allowed_time = 300,
    n_warmup = 50,
    ub = 1e9,
)
    (seed >= 0) && Random.seed!(seed)

    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)

    # Solve
    normal_sddp = NormalSDDP(primal_sddp, mixing, "Normal Solution SDDP [van Ackooij et al. (2019)]")
    primal_models = solve!(normal_sddp, hdm, V,  x₀; n_iter=n_iter, n_cycle=n_cycle, verbose=verbose, τ=τ, n_pruning = n_pruning, allowed_time=allowed_time, n_warmup = n_warmup, upper_bound = ub)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀),
    )
end