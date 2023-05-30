
function warmup!(
    solver::RegularizedPrimalSDDP,
    primal_models,
    dual_models,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    verbose::Int=1,
    n_warming = 50
)
    Ξ = uncertainties(hdm)
    ub = Inf
    for i in 1:n_warming
        scenario = sample(Ξ)
        # Primal
        primal_trajectory = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
        dual_trajectory = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectory, V)
        # Dual
        backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectory, D)
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)

        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
    end
end


function solve2!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    n_cycle = 20,
    verbose::Int=1,
    τ=1e8,
    n_prunning = 100,
    allowed_time = 300,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)
    dual_models = build_stage_models(solver.dual_sddp, hdm, D)

    # Warmup
    tic = time()
    warmup!(solver, primal_models, dual_models, hdm, V, D, x₀; verbose=verbose, n_warming = 50)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Run
    ub = Inf
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    primal_trajectories = Array{Matrix{Float64}}(undef, n_prunning)
    dual_trajectories = Array{Matrix{Float64}}(undef, n_prunning)
    for i in 1:n_iter
        scenario = sample(Ξ)
        j = mod(i, n_prunning) + 1 # Current index since last prunning
        if mod(i, n_cycle) == 0 # Update upper bounds via dual sddp
            # Primal
            primal_trajectories[j] = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
            backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            # Dual
            dual_trajectories[j] = forward_pass(solver.dual_sddp, hdm, dual_models, scenario, p₀)
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        else # Update upper bounds using primal cuts = dual trajectory
            # Primal
            primal_trajectories[j] = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
            dual_trajectories[j] = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            # Dual
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        end
        if  j == n_prunning
            V_ref = V
            D_ref = D
            V = [PolyhedralFunction(length(x₀), V[1](x₀)) for t in 1:length(V)]
            D = [PolyhedralFunction(length(x₀), D[1](dual_trajectories[j][:, 1])) for t in 1:length(D)]
            for j in length(primal_trajectories)
                prunning!(V, V_ref, primal_trajectories[j])
                prunning!(D, D_ref, dual_trajectories[j])
            end
        end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
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
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    return (primal_models, dual_models)
end

# Helper function
function regularizedsddp2(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    n_cycle= 10,
    n_prunning = 100,
    allowed_time = 300,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing)
    primal_models, dual_models = solve2!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, n_cycle=n_cycle, verbose=verbose, τ=τ, n_prunning = n_prunning, allowed_time=allowed_time)

    # Get upper-bound
    ub, _ = fenchel_transform(dual_sddp, D[1], x₀)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀),
        dual_cuts=D,
        dual_models=dual_models,
        upper_bound=ub,
    )
end

function wellington(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    n_cycle= 10,
    n_prunning = 100,
    allowed_time = 300,
    n_scenarios = n_scenarios
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing)
    primal_models, dual_models = solve_wellington!(reg_sddp, hdm, V, x₀; n_iter=n_iter, n_cycle=n_cycle, verbose=verbose, τ=τ, n_prunning = n_prunning, allowed_time=allowed_time, n_scenarios = n_scenarios)


    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀)
    )
end

function solve_wellington!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    n_cycle = 20,
    verbose::Int=1,
    τ=1e8,
    n_prunning = 100,
    allowed_time = 300,
    n_scenarios = 1000,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)

    tic = time()

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Run
    primal_trajectories = Array{Matrix{Float64}}(undef, n_prunning)
    ub = .0
    for i in 1:n_iter
        ub =  montecarlo(solver.primal_sddp, hdm, primal_models, n_scenarios, x₀, Ξ) 
        scenario = sample(Ξ)
        j = mod(i, n_prunning) + 1 # Current index since last prunning
        # Primal
        primal_trajectories[j] = wellington_forward_pass(solver, hdm, primal_models, ub,  V,  scenario, x₀, τ)
        backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
        # if  j == n_prunning
        #     V_ref = V
        #     V = [PolyhedralFunction(length(x₀), V[1](x₀)) for t in 1:length(V)]
        #     for j in length(primal_trajectories)
        #         prunning!(V, V_ref, primal_trajectories[j])
        #     end
        # end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        # Check if allowed time is over
        if time() - tic > allowed_time
            break
        end
    end
    sample
    status = MOI.ITERATION_LIMIT
    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %10s\n\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    return primal_models
end

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

# van Ackooij and al. forward pass 
function wellington_forward_pass!(
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
        # Regularization level ; Adaptative combination between lb and ub_t depending on the relative gap
        relative_gap = (ub_t - lb)/abs(lb)
        mixing = min(relative_gap, 1) # If gap is too big, favor the lb from classic SDDP
        #mixing = .5
        ℓ = mixing * lb + (1.0 - mixing) * ub_t
        
        model = stage_model(hdm, t)
        xₜ = next!(Regsddp, model, V, xₜ, xi, ℓ, τ, t, horizon(hdm))
        cum_cost += stage_objective_value(Regsddp.primal_sddp, model, hdm, t)
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end

function wellington_forward_pass(
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
    return wellington_forward_pass!(Regsddp, hdm, primal_models, init_ub, V, uncertainty_scenario, initial_state, primal_trajectory, τ)
end

function minub(
    sddp::SDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    n_scenarios::Int,
    initial_state::Vector{Float64},
    Ξ::Vector{DiscreteRandomVariable{Float64}},
    t::Int
)
    z = zeros(Float64, length(Ξ[t]))
    for j in 1:length(Ξ[t])
        x_j = next!(sddp, models[t], initial_state, Ξ[t], ξ)
        z[j] = montecarlo(sddp, hdm, models[t+1:horizon(hdm)], n_scenarios, x_j, Ξ[t+1:T], t)
    end
    return minimum(z)
end

function minub_forward_pass!(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
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

        # Upper-bound.
        ubmodel = stage_model(hdm, t)
        
        if t < horizon(hdm)
            ub = init_ub - cum_cost
        else
            ub = lb
        end
        # Regularization level ; Adaptative combination between lb and ub depending on the relative gap
        relative_gap = abs((ub - lb)/lb)
        mixing = min(0.1*relative_gap, 1) # If gap is too big, favor the lb from classic SDDP
        #mixing = .5
        ℓ = mixing * lb + (1.0 - mixing) * ub
        
        model = stage_model(hdm, t)
        xₜ = next!(Regsddp, model, V, xₜ, xi, ℓ, τ, t, horizon(hdm))
        cum_cost += stage_objective_value(Regsddp.primal_sddp, model, hdm, t)
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end