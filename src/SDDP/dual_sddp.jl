#=
    Dual SDDP
=#

struct DualSDDP <: AbstractSDDP
    optimizer::Any
    valid_statuses::Vector{MOI.TerminationStatusCode}
    lipschitz_lb::Float64
    lipschitz_ub::Float64
end

DualSDDP(optimizer; lip_lb=-1e10, lip_ub=1e10) = DualSDDP(optimizer, [MOI.OPTIMAL], lip_lb, lip_ub)

introduce(::DualSDDP) = "Dual SDDP"

#=
    One-stage problem
=#

"""
    _next_costate_reference(model::JuMP.Model, k::Int)

Return the outgoing dual state for the realisation `k` of the 
exogeneous noise.
"""

function _next_costate_reference(model::JuMP.Model, k::Int)
    nx = length(model[_PREVIOUS_COSTATE])
    costates = model[_CURRENT_COSTATE]
    f, t = (k-1) * nx + 1, k * nx
    return costates[f:t]
end

"""
    initialize!(::DualSDDP, stage::Stage, ξₜ₊₁::DiscreteRandomVariable{Float64}, Dₜ₊₁::PolyhedralFunction)

Initialize a stage of the DualSDDP algorithm.

This function sets up the optimization model for a specific stage of the DualSDDP algorithm, incorporating the costate variables, constraints, and objective necessary for the optimization process.

## Arguments
- `::DualSDDP`: The DualSDDP algorithm object.
- `stage::Stage`: The stage for which the optimization model is being initialized.
- `ξₜ₊₁::DiscreteRandomVariable{Float64}`: The random variable representing uncertainty at the next time step.
- `Dₜ₊₁::PolyhedralFunction`: The polyhedral function representing the dual cost-to-go at the next time step.
"""
function initialize!(::DualSDDP, stage::Stage, ξₜ₊₁::DiscreteRandomVariable{Float64}, Dₜ₊₁::PolyhedralFunction)
    nw = length(ξₜ₊₁)
    π = ξₜ₊₁.weights
    @variable(stage.model, θ[1:nw])
    for k in 1:nw
        μk = _next_costate_reference(stage.model, k)
        for (λ, γ) in eachcut(Dₜ₊₁)
            @constraint(stage.model, θ[k] >= λ' * μk + γ)
        end
    end
    obj_expr = objective_function(stage.model)
    @objective(stage.model, Min, -obj_expr + sum(π[k] * θ[k] for k in 1:nw))
    return
end

"""
    solve_stage_problem!(sddp::DualSDDP, stage::Stage, μₜ::Vector{Float64})

Solve the optimization subproblem associated with a single stage in the DualSDDP algorithm.

This function performs the optimization of a specific stage within the DualSDDP algorithm. It fixes the previous costate variables, optimizes the stage's model, and checks the termination status of the solver. If the solver's termination status indicates failure, an error is raised.

## Arguments
- `sddp::DualSDDP`: The DualSDDP algorithm object.
- `stage::Stage`: The stage for which the optimization subproblem is being solved.
- `μₜ::Vector{Float64}`: The vector of fixed values for the previous costate variables.
"""
function solve_stage_problem!(sddp::DualSDDP, stage::Stage, μₜ::Vector{Float64})
    fix.(stage.model[_PREVIOUS_COSTATE], μₜ)
    optimize!(stage.model)
    status = termination_status(stage.model)
    if termination_status(stage.model) ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve dual subproblem: solver's return status is $(termination_status(stage.model))")
    end
    return
end

"""
    fetch_cut(sddp::DualSDDP, model::JuMP.Model)

Return the slope of the cut obtained when solving the current model.
#TODO: unclear
"""
fetch_cut(sddp::DualSDDP, model::JuMP.Model) = dual.(FixRef.(model[_PREVIOUS_COSTATE]))

"""
    add_dual_cut!(stage::Stage, Dₜ::PolyhedralFunction, x, γ)

    Add a cut with slope `x` and offset `γ` for the dual value function of the current stage.

    Add the cut both to the PolyhedralFunction Dₜ and to the parent `model`
"""
function add_dual_cut!(stage::Stage, Dₜ::PolyhedralFunction, x, γ)
    add_cut!(Dₜ, x, γ)

    parent = stage.parent
    if !isnothing(parent)
        nw = length(parent.model[_VALUE_FUNCTION])
        for k in 1:nw
            μk = _next_costate_reference(parent.model, k)
            @constraint(parent.model, parent.model[_VALUE_FUNCTION][k] >= x' * μk + γ)
        end
    end
    return
end

"""
    next!(
        sddp::DualSDDP,
        stage::Stage,
        μₜ::Vector{Float64},
        ξ::DiscreteRandomVariable{Float64},
        ξₜ₊₁::Vector{Float64}
    )

Advance to the next stage of the Dual Stochastic Dynamic Programming (DualSDDP) algorithm.

This function progresses the DualSDDP algorithm to the next time step by performing the following steps:
1. Solve the optimization subproblem for the current stage using fixed previous costate variables.
2. Determine the outcome index `k` based on the realization of the random variable `ξₜ₊₁`.
3. Retrieve the costate variables associated with the determined outcome `k` using `_next_costate_reference`.

## Arguments
- `sddp::DualSDDP`: The DualSDDP algorithm object.
- `stage::Stage`: The current stage of the algorithm.
- `μₜ::Vector{Float64}`: The previous costate variables.
- `ξ::DiscreteRandomVariable{Float64}`: The random variable representing uncertainty at the current time step.
- `ξₜ₊₁::Vector{Float64}`: The realization of the uncertainty variable considered.

## Returns
Next optimal costate
"""
function next!(
    sddp::DualSDDP,
    stage::Stage,
    μₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
)
    solve_stage_problem!(sddp, stage, μₜ)
    k = find_outcome(ξ, ξₜ₊₁)
    @assert 1 <= k <= length(ξ)
    μf = _next_costate_reference(stage.model, k)
    return JuMP.value.(μf)
end

"""
    previous!(
        sddp::DualSDDP,
        stage::Stage,
        μₜ::Vector{Float64},
        ξ::DiscreteRandomVariable{Float64},
        Dₜ::PolyhedralFunction
    )

Perform the backward pass of the DualSDDP algorithm.

This function performs the backward pass of the DualSDDP algorithm for a specific stage. 
    It solves the optimization subproblem, fetches the dual values associated with fixed previous costate variables, computes a dual cut, and update the polyhedral models.

## Arguments
- `sddp::DualSDDP`: The DualSDDP algorithm object.
- `stage::Stage`: The current stage of the algorithm.
- `μₜ::Vector{Float64}`: The vector of fixed values for the previous costate variables.
- `ξ::DiscreteRandomVariable{Float64}`: The random variable representing uncertainty at the current time step.
- `Dₜ::PolyhedralFunction`: The polyhedral function representing the dual cost-to-go at the current time step.

## Returns
The slope of the dual cut which can be interpreted as a primal state.
"""
function previous!(
    sddp::DualSDDP,
    stage::Stage,
    μₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    Dₜ::PolyhedralFunction,
)
    nx = length(μₜ)
    solve_stage_problem!(sddp, stage, μₜ)
    x = fetch_cut(sddp, stage.model)
    γ = objective_value(stage.model) - dot(x, μₜ)
    add_dual_cut!(stage, Dₜ, x, γ)
    return x
end


#=
    Algorithm
=#

"""
    fenchel_transform(solver::DualSDDP, D::PolyhedralFunction, x)

Compute the Fenchel value D*(x) and subgradient λ∈∂D(x) #TODO:Check

## Arguments
- `solver::DualSDDP`: The DualSDDP algorithm object.
- `D::PolyhedralFunction`: The polyhedral function for which the Fenchel dual is computed.
- `x`: The point for which the Fenchel dual is calculated.

## Returns
A tuple `(value, λ)` where:
- `value`: The value of the Fenchel dual function.
- `λ`: subgradient.
"""
function fenchel_transform(solver::DualSDDP, D::PolyhedralFunction, x)
    nx = dimension(D)
    model = Model()
    @variable(model, solver.lipschitz_lb <= λ[1:nx] <= solver.lipschitz_ub)
    @variable(model, θ)
    for (xk, βk) in eachcut(D)
        @constraint(model, θ >= dot(xk, λ) + βk)
    end
    @objective(model, Max, dot(x, λ) - θ)

    JuMP.set_optimizer(model, solver.optimizer)
    JuMP.optimize!(model)

    return JuMP.objective_value(model), JuMP.value.(λ)
end

"""
    build_tree(solver::DualSDDP, hdm::HazardDecisionModel, D::Vector{PolyhedralFunction})

Build a multistage tree of optimization models for the DualSDDP algorithm.

This function constructs a multistage tree of optimization models based on the given hazard decision model and vector of polyhedral functions. The tree represents the stages and optimization problems used in the DualSDDP algorithm.
"""
function build_tree(solver::DualSDDP, hdm::HazardDecisionModel, D::Vector{PolyhedralFunction})
    Ξ = uncertainties(hdm)
    T = horizon(hdm)
    # Initialize model between time t=1 up to T-1
    parent = nothing
    stages = Stage{JuMP.Model}[]
    for t in 1:T
        lb, ub = if t < T
            solver.lipschitz_lb, solver.lipschitz_ub
        else
            # Set final co-state (μ₊ = 0) to let Dualization.jl take care of final costs.
            0.0, 0.0
        end
        pb = dual_stage_model(hdm, t, lb, ub)
        stage = Stage(parent, pb, t, t==T)
        push!(stages, stage)
        parent = stage
    end

    for stage in stages
        if stage.t < T
            initialize!(solver, stage, Ξ[stage.t], D[stage.t+1])
        else
            obj_expr = objective_function(stage.model)
            @objective(stage.model, Min, -obj_expr)
        end
        JuMP.set_optimizer(stage.model, solver.optimizer)
    end

    return MultistageProblem(hdm, stages)
end

#=
    Forward pass for dual SDDP (aka CUPPS)
=#
function forward_pass!(
    sddp::DualSDDP,
    tree::MultistageProblem,
    scenario::InSampleScenario{Float64},
    initial_state::Vector{Float64},
    V::Vector{PolyhedralFunction},
)
    Ξ = uncertainties(tree.model)
    nx, T = number_states(tree.model), horizon(tree.model)
    trajectory = fill(0.0, nx, T + 1)
    trajectory[:, 1] .= initial_state
    for stage in tree.stages
        xₜ = trajectory[:, stage.t]
        wₜ = scenario.values[:, stage.t]
        trajectory[:, stage.t+1] .= next!(sddp, stage, xₜ, Ξ[stage.t], wₜ)
        # Fetch cut and add it directly.
        λ = fetch_cut(sddp, stage.model)
        γ = JuMP.objective_value(stage.model) - dot(λ, trajectory[:, stage.t])
        add_dual_cut!(stage, V[stage.t], λ, γ)
    end
    return trajectory
end

function solve!(
    solver::DualSDDP,
    hdm::HazardDecisionModel,
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int = 1,
)
    (verbose > 0) && header()

    tree = build_tree(solver, hdm, D)
    Ξ = uncertainties(hdm)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s\n", "-"^4, "-"^15)
        @printf(" %4s %15s\n", "#it", "UB")
    end

    # Run
    tic = time()
    ub, p₀ = fenchel_transform(solver, D[1], x₀)
    for i in 1:n_iter
        scenario = sample(Ξ)
        dual_trajectory = forward_pass!(solver, tree, scenario, p₀, D)
        primal_trajectory = backward_pass!(solver, tree, dual_trajectory, D)
        ub, p₀ = fenchel_transform(solver, D[1], x₀)
        if (verbose > 0) && (mod(i, verbose) == 0)
            @printf(" %4i %15.6e\n", i, ub)
        end
    end

    if verbose > 0
        @printf(" %4s %15s\n\n", "-"^4, "-"^15)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Upper-bound.....: %15.8e\n", ub)
    end

    return tree
end

# Helper function
function dualsddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    seed=0,
    n_iter=500,
    verbose::Int = 1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)
    dual_models = solve!(dual_sddp, hdm, D, x₀; n_iter=n_iter, verbose=verbose)
    ub, _ = fenchel_transform(dual_sddp, D[1], x₀)
    return (cuts=D, models=dual_models, upper_bound=ub)
end

