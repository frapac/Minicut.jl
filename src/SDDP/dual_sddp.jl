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

function _next_costate_reference(model::JuMP.Model, k::Int)
    nx = length(model[_PREVIOUS_COSTATE])
    costates = model[_CURRENT_COSTATE]
    f, t = (k-1) * nx + 1, k * nx
    return costates[f:t]
end

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

function solve_stage_problem!(sddp::DualSDDP, stage::Stage, μₜ::Vector{Float64})
    fix.(stage.model[_PREVIOUS_COSTATE], μₜ)
    optimize!(stage.model)
    status = termination_status(stage.model)
    if termination_status(stage.model) ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve dual subproblem: solver's return status is $(termination_status(stage.model))")
    end
    return
end

fetch_cut(sddp::DualSDDP, model::JuMP.Model) = dual.(FixRef.(model[_PREVIOUS_COSTATE]))

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
    saving_data::Bool=saving_data
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
    if saving_data
        df = init_save(hdm, n_iter)
    end

    # Run
    tic = time()
    ub, p₀ = fenchel_transform(solver, D[1], x₀)
    for i in 1:n_iter
        scenario = sample(Ξ)
        if saving_data 
            df.timers[i, :time_dual_forward] = @elapsed(dual_trajectory = forward_pass!(solver, tree, scenario, p₀, D))
            df.timers[i, :time_dual_backward] = @elapsed(primal_trajectory = backward_pass!(solver, tree, dual_trajectory, D))
            # UB at (x0, x_1^k,...,x_T^k) where x0 is given and x_1^k is the primal traj. given by cuts in the dual
            df.ub[i, 2] = fenchel_transform(solver, D[1], x₀)[1]
            for t in 2:horizon(hdm)
                df.ub[i, t+1] = fenchel_transform(solver, D[t], primal_trajectory[:, t])[1]
            end
            if i in [200,250,300]
                save("D_$(i).jld2", Dict("D"=>D))
            end
        else
            dual_trajectory = forward_pass!(solver, tree, scenario, p₀, D)
            primal_trajectory = backward_pass!(solver, tree, dual_trajectory, D)
        end
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

    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_dualsddp_data.csv", df.data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_dualsddp_timers.csv", df.timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_dualsddp_ub.csv", df.ub) 
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
    saving_data::Bool=false
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)
    dual_models = solve!(dual_sddp, hdm, D, x₀; n_iter=n_iter, verbose=verbose, saving_data = saving_data)
    ub, _ = fenchel_transform(dual_sddp, D[1], x₀)
    return (cuts=D, models=dual_models, upper_bound=ub)
end

