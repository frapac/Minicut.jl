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

function initialize!(::DualSDDP, model::JuMP.Model, ξₜ₊₁::DiscreteRandomVariable{Float64}, Dₜ₊₁::PolyhedralFunction)
    nw = length(ξₜ₊₁)
    π = ξₜ₊₁.weights
    @variable(model, θ[1:nw])
    for k in 1:nw
        μk = _next_costate_reference(model, k)
        for (λ, γ) in eachcut(Dₜ₊₁)
            @constraint(model, θ[k] >= λ' * μk + γ)
        end
    end
    obj_expr = objective_function(model)
    @objective(model, Min, -obj_expr + sum(π[k] * θ[k] for k in 1:nw))
    return
end

function solve_stage_problem!(sddp::DualSDDP, model::JuMP.Model, μₜ::Vector{Float64})
    fix.(model[_PREVIOUS_COSTATE], μₜ)
    optimize!(model)
    v = JuMP.all_variables(model)
    if termination_status(model) ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve dual subproblem: solver's return status is $(termination_status(model))")
    end
    return
end

fetch_cut(sddp::DualSDDP, model::JuMP.Model) = dual.(FixRef.(model[_PREVIOUS_COSTATE]))

function next!(
    sddp::DualSDDP,
    model::JuMP.Model,
    μₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
)
    solve_stage_problem!(sddp, model, μₜ)
    k = find_outcome(ξ, ξₜ₊₁)
    @assert 1 <= k <= length(ξ)
    μf = _next_costate_reference(model, k)
    return JuMP.value.(μf)
end

function previous!(
    sddp::DualSDDP,
    model::JuMP.Model,
    μₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    Dₜ::PolyhedralFunction,
)
    nx = length(μₜ)
    solve_stage_problem!(sddp, model, μₜ)
    x = fetch_cut(sddp, model)
    γ = objective_value(model) - dot(x, μₜ)
    add_cut!(Dₜ, x, γ)
    return x
end

function synchronize!(::DualSDDP, model::JuMP.Model, Dₜ₊₁::PolyhedralFunction)
    nw = length(model[_VALUE_FUNCTION])
    for k in 1:nw
        μk = _next_costate_reference(model, k)
        @constraint(model, model[_VALUE_FUNCTION][k] >= Dₜ₊₁.λ[end, :]' * μk + Dₜ₊₁.γ[end])
    end
    return
end

#=
    Algorithm
=#

function fenchel_transform(solver::DualSDDP, D::PolyhedralFunction, x)
    nx = dimension(D)
    model = Model()
    #@variable(model, solver.lipschitz_lb <= λ[1:nx] <= 0.0) # Why upper bounded by 0? Isn't it norm(λ, Inf) < L_t theoritically?
    @variable(model, solver.lipschitz_lb ≤ λ[1:nx] ≤ solver.lipschitz_ub)
    @variable(model, θ)
    for (xk, βk) in eachcut(D)
        @constraint(model, θ >= dot(xk, λ) + βk)
    end
    @objective(model, Max, dot(x, λ) - θ)
    JuMP.set_optimizer(model, solver.optimizer)
    JuMP.optimize!(model)

    return JuMP.objective_value(model), JuMP.value.(λ)
end

function build_stage_models(solver::DualSDDP, hdm::HazardDecisionModel, D::Vector{PolyhedralFunction})
    Ξ = uncertainties(hdm)
    T = horizon(hdm)
    # Initialize model between time t=1 up to T-1
    models = [dual_stage_model(hdm, t, solver.lipschitz_lb, solver.lipschitz_ub) for t in 1:T-1]
    # NB: we define apart the model for final stage and we deactivate
    #     final co-state (μ₊ = 0) to let Dualization.jl take care of final costs.
    push!(models, dual_stage_model(hdm, T, 0.0, 0.0))
    for (t, model) in enumerate(models)
        if t < T
            initialize!(solver, model, Ξ[t], D[t+1])
        else
            obj_expr = objective_function(model)
            @objective(model, Min, -obj_expr)
        end
        JuMP.set_optimizer(model, solver.optimizer)
    end
    return models
end

function solve!(
    solver::DualSDDP,
    hdm::HazardDecisionModel,
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int = 1,
    allowed_time = 1200,
    saving_data = false,
)
    (verbose > 0) && header()

    models = build_stage_models(solver, hdm, D)
    Ξ = uncertainties(hdm)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s\n", "-"^4, "-"^15)
        @printf(" %4s %15s\n", "#it", "UB")
    end

    run_data, run_timers, run_ub = init_data(
    solver,
    models,
    hdm,
    D,
    x₀,
    allowed_time,
    n_iter,
    )

    # Run
    time_mainrun = time()
    ub, p₀ = fenchel_transform(solver, D[1], x₀)
    for i in 1:n_iter
        tic_iter = time()
        scenario = sample(Ξ)
        tic = time()
        dual_trajectory = cupps_pass!(solver, hdm, models, scenario, p₀, D)
        run_timers[i, :time_dual_forward] += time() - tic 
        tic = time()
        primal_trajectory = backward_pass!(solver, hdm, models, dual_trajectory, D)
        run_timers[i, :time_dual_backward] += time() - tic
        ub, p₀ = fenchel_transform(solver, D[1], x₀)
        run_timers[i, :time_iter] += time() - tic_iter
        if (verbose > 0) && (mod(i, verbose) == 0)
            @printf(" %4i %15.6e\n", i, ub)
        end
        if time() - time_mainrun > allowed_time
            break
        end
        if saving_data
            for t in 1:horizon(hdm)
                run_ub[i,t+1] = fenchel_transform(solver, D[t], dual_trajectory[:, t])[1]
            end
        end
    end

    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_rundata_dualsddp.csv", run_data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runtimers_dualsddp.csv", run_timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runub_dualsddp.csv", run_ub) 
    end 

    if verbose > 0
        @printf(" %4s %15s\n\n", "-"^4, "-"^15)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - time_mainrun)
        @printf("Upper-bound.....: %15.8e\n", ub)
    end

    return models
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
    allowed_time = 1200,
    saving_data = false,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)
    dual_models = solve!(dual_sddp, hdm, D, x₀; n_iter=n_iter, verbose=verbose, allowed_time = allowed_time, saving_data = saving_data)
    ub, _ = fenchel_transform(dual_sddp, D[1], x₀)
    return (cuts=D, models=dual_models, upper_bound=ub)
end

