
#=
    Add custom structure to handle normalized
    multistage problem.
=#

struct NormalizedStage{M} <: AbstractNode
    parent::Union{Nothing, NormalizedStage{M}}
    model::M
    regularized_model::M
    t::Int
    is_final::Bool
    # Information for regularization
    paths_to_costs::Dict{UInt64, Vector{Float64}}
    nodes_to_paths::Dict{Int, Vector{UInt64}}
    upperbounds::Vector{Float64}
    cum_costs::Vector{Float64}
end

function reset!(stage::NormalizedStage)
    empty!(stage.paths_to_costs)
    empty!(stage.nodes_to_paths)
    return
end

function _update_scenarios!(
    tree::MultistageProblem,
    scenarios::Vector{InSampleScenario{Float64}},
    cum_costs::Matrix{Float64},
)
    # Scan scenarios.
    for (k, scen) in enumerate(scenarios)
        for stage in tree.stages
            # Get ID of the current path using its hash.
            id_path = hash(scen.path[1:stage.t])

            # Add remaining cost to previous_paths
            expected_cost = cum_costs[k, end] - cum_costs[k, stage.t]
            if haskey(stage.paths_to_costs, id_path)
                push!(stage.paths_to_costs[id_path], expected_cost)
            else
                stage.paths_to_costs[id_path] = [expected_cost]
            end

            # Store a reference to the current path.
            node = scen.path[stage.t]
            if haskey(stage.nodes_to_paths, node)
                push!(stage.nodes_to_paths[node], id_path)
            else
                stage.nodes_to_paths[node] = [id_path]
            end
        end
    end
end

function _update_upperbounds!(
    tree::MultistageProblem,
)
    for stage in tree.stages
        n_nodes = length(stage.upperbounds)
        for k in 1:n_nodes
            if !haskey(stage.nodes_to_paths, k)
                continue
            end
            for path in stage.nodes_to_paths[k]
                costs_ = stage.paths_to_costs[path]
                stage.upperbounds[k] = min(stage.upperbounds[k], sum(costs_) / length(costs_))
            end
        end
    end
end


struct NormalSDDP <: AbstractSDDP
    qp_optimizer::Any
    primal_sddp::SDDP
    tau::Float64
    max_upperbound::Float64
    n_forward::Int
end

NormalSDDP(optimizer, sddp::SDDP; tau=1e8, max_ub=1e10, n_forward=10) = NormalSDDP(optimizer, sddp, tau, max_ub, n_forward)
introduce(solver::NormalSDDP) = "Normal SDDP"

function initialize!(sddp::NormalSDDP, stage::NormalizedStage, Vₜ₊₁::PolyhedralFunction)
    # Original model
    @variable(stage.model, θ)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(stage.model, θ >= λ' * stage.model[_CURRENT_STATE] + γ)
    end
    obj_expr = JuMP.objective_function(stage.model)
    @objective(stage.model, Min, obj_expr + θ)

    # Regularized model
    @expression(stage.regularized_model, cost, JuMP.objective_function(stage.regularized_model))
    @variable(stage.regularized_model, θ)
    @variable(stage.regularized_model, w)
    @variable(stage.regularized_model, level)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(stage.regularized_model, θ >= λ' * stage.regularized_model[_CURRENT_STATE] + γ)
    end
    @constraint(stage.regularized_model, w >= θ + cost)
    @constraint(stage.regularized_model, w >= level)
    @objective(stage.regularized_model, Min, w + (1 / (2 * sddp.tau)) * sum(stage.regularized_model[_CURRENT_STATE]).^2)

    return
end

function stage_objective_value(sddp::NormalSDDP, stage::NormalizedStage)
    if stage.is_final
        return JuMP.objective_value(stage.regularized_model)
    else
        return JuMP.value(stage.regularized_model[:cost])
    end
end

function add_cut!(stage::NormalizedStage, Vₜ::PolyhedralFunction, λ, γ)
    add_cut!(Vₜ, λ, γ)

    parent = stage.parent
    if !isnothing(parent)
        # Add cut both in original and regularized models.
        for model in [parent.model, parent.regularized_model]
            @constraint(model, model[_VALUE_FUNCTION] >= λ' * model[_CURRENT_STATE] + γ)
        end
    end
    return
end

function next!(
    sddp::NormalSDDP,
    stage::NormalizedStage,
    xₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
    level::Float64,
)
    fix.(stage.regularized_model[_PREVIOUS_STATE], xₜ, force = true)
    fix.(stage.regularized_model[_UNCERTAINTIES], ξₜ₊₁, force = true)
    if !stage.is_final
        fix.(stage.regularized_model[_LEVEL], level, force = true)
    end
    optimize!(stage.regularized_model)
    status = termination_status(stage.regularized_model)
    if status ∉ sddp.primal_sddp.valid_statuses
        error("[SDDP] Fail to solve primal subproblem: solver's return status is $(status).")
    end
    return JuMP.value.(stage.regularized_model[_CURRENT_STATE])
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
    forward_pass!(sddp, tree, scenario, initial_state, primal_trajectory, gap)
    return primal_trajectory
end

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
    for stage in tree.stages
        wₜ = scenarios.values[:, stage.t]
        j = scenarios.path[stage.t]
        gamma = 0.5 / stage.t
        level = stage.upperbounds[j] - gamma * max(0.0, gap)
        xₜ = next!(sddp, stage, xₜ, Ξ[stage.t], wₜ, level)
        trajectory[:, stage.t+1] .= xₜ
    end
    return trajectory
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

    for (k, scenario) in enumerate(scenarios)
        push!(trajectories, forward_pass(sddp, tree, scenario, initial_state, gap))
        # Update cum_costs
        for stage in tree.stages
            cost = stage_objective_value(sddp, stage)
            cum_costs[k, stage.t+1] = cum_costs[k, stage.t] + cost
        end
    end

    for stage in tree.stages
        reset!(stage)
    end
    _update_scenarios!(tree, scenarios, cum_costs)
    _update_upperbounds!(tree)
    return trajectories, cum_costs
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
        nodes_to_paths = Dict{Int, Vector{UInt64}}()

        npb = stage_model(hdm, t) # original problem
        rpb = stage_model(hdm, t) # regularized problem
        is_final = (t == T)
        stage = NormalizedStage(
            parent, npb, rpb, t, is_final,
            paths_to_costs, nodes_to_paths,
            zeros(n_nodes[t]),
            zeros(solver.n_forward),
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
        JuMP.set_attribute(stage.regularized_model, "OptimalityTol", 1e-4)
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
        @printf(" %4s %15s\n", "-"^4, "-"^15)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "gap")
    end

    ub = Inf
    gap = solver.max_upperbound

    tic = time()
    # Run
    for i in 1:n_iter
        scenario = sample(Ξ, solver.n_forward)
        primal_trajectory, cum_costs = forward_pass(solver, normal_tree, scenario, x₀, gap)
        backward_pass!(solver.primal_sddp, normal_tree, primal_trajectory, V)

        lb = V[1](x₀)
        ub = sum(cum_costs[:, end]) / solver.n_forward
        gap = (ub - lb)
        if (verbose > 0) && (mod(i, verbose) == 0)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100*gap/abs(lb))
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

