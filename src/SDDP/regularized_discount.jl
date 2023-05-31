#=
    Regularized SDDP with the Discount rule to compute the upperbounds [van Ackooij et al (2019)]
=#

function solve_discount!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int=1,
    τ=1e8,
    n_pruning = 100,
    allowed_time = 300,
    n_scenarios = 1000,
    n_warmup = 50,
)
    (verbose > 0) && header()
    
    Ξ = uncertainties(hdm)
    primal_models = build_stage_models(solver.primal_sddp, hdm, V)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
    end

    # Warmup
    if n_warmup > 0
        println("Warming up")
        V = warmup!(solver.primal_sddp, primal_models, hdm, V, x₀; verbose=verbose, n_warmup = n_warmup)
    end

    if verbose > 0
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Run
    tic = time()
    primal_trajectories = Array{Matrix{Float64}}(undef, n_pruning)
    ub = .0
    for i in 1:n_iter
        j = mod(i, n_pruning) + 1 # Current index since last pruning
        ub =  montecarlo(solver.primal_sddp, hdm, primal_models, n_scenarios, x₀, Ξ) 
        scenario = sample(Ξ)
        # Primal
        primal_trajectories[j] = discount_forward_pass(solver, hdm, primal_models, ub,  V,  scenario, x₀, τ)
        backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        if  j == 1
            V = pruning(V, primal_trajectories)
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
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    return primal_models
end

function discount_forward_pass!(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    init_ub::Float64,
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
    τ::Float64
)
    Ξ = uncertainties(hdm)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    cum_cost = 0.0

    for (t, ξₜ₊₁) in enumerate(eachcol(uncertainty_scenario))
        xi = collect(ξₜ₊₁)
        # Lower-bound.
        lb = lowerbound(Regsddp.primal_sddp, primal_models[t], xₜ, xi)
     
        if t == 1 
            ub_t = init_ub
        elseif t < horizon(hdm)
            ub_t = init_ub - cum_cost
        else
            ub_t = lb
        end
        # RegularizationD level ; Adaptative combination between lb and ub_t depending on the relative gap
        γₜ = 0.5 / t
        mixing = γₜ*lb  + (1-γₜ)*ub_t
        ℓ = mixing * lb + (1.0 - mixing) * ub_t
        
        model = stage_model(hdm, t)
        xₜ = next!(Regsddp, model, V, xₜ, xi, ℓ, τ, t, horizon(hdm))
        cum_cost += stage_objective_value(Regsddp.primal_sddp, model, hdm, t)
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end

function discount_forward_pass(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    init_ub::Float64,
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    τ::Float64
)
    horizon = size(uncertainty_scenario, 2)
    primal_trajectory = fill(0.0, length(initial_state), horizon + 1)
    return discount_forward_pass!(Regsddp, hdm, primal_models, init_ub, V, uncertainty_scenario, initial_state, primal_trajectory, τ)
end

function reg_discount(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer,
    V::Array{PolyhedralFunction};
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    n_pruning = 100,
    allowed_time = 300,
    n_scenarios = 1000,
    n_warmup = n_warmup
)
    (seed >= 0) && Random.seed!(seed)

    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing, "Regularized SDDP with Discount rule (van Ackooij et al. (2019))") # useless dual model, to change
    primal_models = solve_discount!(reg_sddp, hdm, V, x₀; n_iter=n_iter, verbose=verbose, τ=τ, n_pruning = n_pruning, allowed_time = allowed_time, n_scenarios = n_scenarios, n_warmup = n_warmup)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀)
    )
end



######### AUXILIARY FUNCTIONS

function montecarlo(
    sddp::SDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    n_scenarios::Int,
    initial_state::Vector{Float64},
    Ξ::Vector{DiscreteRandomVariable{Float64}}
)
    scenarios = Minicut.sample(Ξ, n_scenarios)
    costs = Minicut.simulate!(sddp, hdm, models, initial_state, scenarios)
    return mean(costs) + 1.96 * std(costs) / sqrt(n_scenarios)
end

function montecarlo(
    sddp::SDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    n_scenarios::Int,
    initial_state::Vector{Float64},
    Ξ::Vector{DiscreteRandomVariable{Float64}},
    t::Int
)
    @assert t < horizon(hdm) 
    scenarios = Minicut.sample(Ξ[t+1:horizon(hdm)], n_scenarios)
    costs = Minicut.simulate!(sddp, hdm, models, initial_state, scenarios, t)
    return mean(costs) + 1.96 * std(costs) / sqrt(n_scenarios)
end

function simulate!(
    sddp::AbstractSDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    initial_state::Vector{Float64},
    uncertainty_scenario::Vector{Array{Float64,2}},
    t0::Int
)
    @assert t0 < horizon(hdm)
    Ξ = uncertainties(hdm)
    n_scenarios = length(uncertainty_scenario)
    n_states = number_states(hdm)
    xₜ = zeros(n_states)
    costs = zeros(n_scenarios)
    for k in 1:n_scenarios
        xₜ .= initial_state
        for t in t0:horizon(hdm)
            ξ = uncertainty_scenario[k][:, t]
            xₜ = next!(sddp, models[t], xₜ, Ξ[t], ξ)
            costs[k] += stage_objective_value(sddp, models[t], hdm, t)
        end
    end
    return costs
end


function warmup!(
    solver::SDDP,
    primal_models,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    verbose::Int=1,
    n_warmup = 50,
    n_pruning = 200,
)
    Ξ = uncertainties(hdm)
    primal_trajectories = Array{Matrix{Float64}}(undef, n_pruning)
    for i in 1:n_warmup
        j = mod(i, n_pruning) + 1
        scenario = sample(Ξ)
        # Primal
        primal_trajectories[j] = forward_pass(solver, hdm, primal_models, scenario, x₀)
        backward_pass!(solver, hdm, primal_models, primal_trajectories[j], V)
        if  j == 1
            V = pruning(V, primal_trajectories)
        end
        if (verbose > 0) && (mod(i, verbose) == 0) && (n_warmup > 0)
            lb = V[1](x₀)
            @printf(" %4i %15.6e \n", i, lb)
        end
    end
    if n_warmup > 0
        println("Mean bundle size after warmup: $(mean(ncuts(V[t]) for t in 1:horizon(hdm)))")
    end
    return V,D
end