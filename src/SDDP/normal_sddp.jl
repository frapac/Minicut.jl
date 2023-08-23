
#=
    Add custom structure to handle normalized multistage problem.
=#

struct NormalizedStage{M} <: AbstractRegularizedNode
    parent::Union{Nothing, NormalizedStage{M}}
    model::M
    regularized_model::M
    t::Int
    is_final::Bool
    # Information for regularization
    paths_to_costs::Dict{UInt64, Vector{Float64}}
    upperbounds::Vector{Float64}
end

function reset!(stage::NormalizedStage)
    empty!(stage.paths_to_costs)
    return
end

function _update_scenarios!(
    tree::MultistageProblem,
    scenarios::Vector{InSampleScenario{Float64}},
    cum_costs::Matrix{Float64},
)
    history = zeros(UInt64, length(scenarios), horizon(tree.model))
    # Scan scenarios.
    for (k, scen) in enumerate(scenarios)
        for stage in tree.stages
            # Get ID of the current path using its hash.
            id_path = hash(scen.path[1:stage.t])
            history[k, stage.t] = id_path

            # Add remaining cost to previous_paths
            expected_cost = cum_costs[k, end] - cum_costs[k, stage.t+1]
            if haskey(stage.paths_to_costs, id_path)
                push!(stage.paths_to_costs[id_path], expected_cost)
            else
                stage.paths_to_costs[id_path] = [expected_cost]
            end
        end
    end
    return history
end

function _update_upperbounds!(
    tree::MultistageProblem,
    scenarios::Vector{InSampleScenario{Float64}},
    cum_costs::Matrix{Float64},
    history::Matrix{UInt64},
)
    for (k, scenario) in enumerate(scenarios)
        for stage in tree.stages
            j = scenario.path[stage.t]
            path = history[k, stage.t]
            current_cost = cum_costs[k, stage.t+1] - cum_costs[k, stage.t]
            future_costs = stage.paths_to_costs[path]
            expected_cost = sum(future_costs) / length(future_costs)
            stage.upperbounds[j] = min(
                stage.upperbounds[j],
                (current_cost + expected_cost),
            )
        end
    end
end


struct NormalSDDP <: AbstractRegularizedSDDP
    qp_optimizer::Any
    primal_sddp::SDDP
    tau::Float64
    max_upperbound::Float64
    n_forward::Int
end

NormalSDDP(optimizer, sddp::SDDP; tau=1e8, max_ub=1e10, n_forward=10) = NormalSDDP(optimizer, sddp, tau, max_ub, n_forward)
introduce(solver::NormalSDDP) = "Normal SDDP"

function forward_pass!(
    sddp::NormalSDDP,
    tree::MultistageProblem,
    scenarios::InSampleScenario{Float64},
    initial_state::Vector{Float64},
    trajectory::Array{Float64, 2},
    gap::Float64,
)
    Ξ = uncertainties(tree.model)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    level_cnt = 0
    for stage in tree.stages
        wₜ = scenarios.values[:, stage.t]
        j = scenarios.path[stage.t]
        gamma = 1.0 #0.5 / stage.t
        level = if !isinf(stage.upperbounds[j])
            stage.upperbounds[j] - gamma * max(0.0, gap)
        else
            -sddp.max_upperbound
        end
        xₜ, is_level = next!(sddp, stage, xₜ, Ξ[stage.t], wₜ, level)
        trajectory[:, stage.t+1] .= xₜ
        level_cnt += is_level
    end
    return trajectory, level_cnt
end

function forward_pass(
    sddp::NormalSDDP,
    tree::MultistageProblem,
    scenario::InSampleScenario{Float64},
    initial_state::Vector{Float64},
    gap::Float64,
)
    T = horizon(tree.model)
    primal_trajectory = fill(0.0, length(initial_state), T + 1)
    return forward_pass!(sddp, tree, scenario, initial_state, primal_trajectory, gap)
end

function forward_pass(
    sddp::NormalSDDP,
    tree::MultistageProblem,
    scenarios::Vector{InSampleScenario{Float64}},
    initial_state::Vector{Float64},
    gap::Float64,
)
    trajectories = Matrix{Float64}[]
    cum_costs = zeros(length(scenarios), horizon(tree.model)+1)

    level_cnt = 0
    for (k, scenario) in enumerate(scenarios)
        traj, cnt = forward_pass(sddp, tree, scenario, initial_state, gap)
        level_cnt += cnt
        push!(trajectories, traj)
        # Update cum_costs
        for stage in tree.stages
            cost = stage_objective_value(sddp, stage)
            cum_costs[k, stage.t+1] = cum_costs[k, stage.t] + cost
        end
    end

    for stage in tree.stages
        reset!(stage)
    end
    history = _update_scenarios!(tree, scenarios, cum_costs)
    _update_upperbounds!(tree, scenarios, cum_costs, history)
    return trajectories, cum_costs, level_cnt
end

function build_tree(solver::NormalSDDP, hdm::HazardDecisionModel, V::Vector{PolyhedralFunction})
    # Get number of nodes per stage.
    n_nodes = length.(uncertainties(hdm))
    T = horizon(hdm)

    # Build multistage tree
    stages = NormalizedStage{JuMP.Model}[]
    parent = nothing
    for t in 1:T
        # Build default multistage problem.
        paths_to_costs = Dict{UInt64, Vector{Float64}}()

        npb = stage_model(hdm, t) # original problem
        rpb = stage_model(hdm, t) # regularized problem
        is_final = (t == T)
        stage = NormalizedStage(
            parent, npb, rpb, t, is_final,
            paths_to_costs,
            fill(Inf, n_nodes[t]),
        )
        push!(stages, stage)
        parent = stage
    end

    # Initialize JuMP model
    for stage in stages
        if !stage.is_final
            initialize!(solver, stage, V[stage.t+1])
        end
        JuMP.set_optimizer(stage.model, solver.primal_sddp.optimizer)
        JuMP.set_optimizer(stage.regularized_model, solver.qp_optimizer)
    end

    return MultistageProblem(hdm, stages)
end

function solve!(
    solver::NormalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int=1,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    normal_tree = build_tree(solver, hdm, V)
    # return normal_tree

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "gap")
    end

    ub = Inf
    gap = solver.max_upperbound

    tic = time()
    # Run
    for i in 1:n_iter
        scenario = sample(Ξ, solver.n_forward)
        # if i <= 10
        #     gap = 1e9
        # end
        primal_trajectory, cum_costs, level_cnt = forward_pass(solver, normal_tree, scenario, x₀, gap)
        backward_pass!(solver.primal_sddp, normal_tree, primal_trajectory, V)

        lb = V[1](x₀)
        ub = sum(cum_costs[:, end]) / solver.n_forward
        gap = (ub - lb)
        if (verbose > 0) && (mod(i, verbose) == 0)
            @printf(" %4i %15.6e %15.6e %10.3f %3i\n", i, lb, ub, 100*gap/abs(lb), level_cnt)
        end
    end

    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %10s\n\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
    end
    return normal_tree
end

function normalsddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer_lp,
    optimizer_qp;
    τ=1e8,
    seed=0,
    n_iter=500,
    n_forward = 10,
    verbose::Int=1,
    lower_bound=-1e6,
    upper_bound=1e10,
    valid_statuses=[MOI.OPTIMAL],
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer_lp, valid_statuses)

    # Solve
    normal_sddp = NormalSDDP(optimizer_qp, primal_sddp, τ, upper_bound, n_forward)
    primal_models = solve!(normal_sddp, hdm, V, x₀; n_iter=n_iter, verbose=verbose)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀)
    )
end
