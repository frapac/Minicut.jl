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

#=
    One-stage problem
=#

function _next_costate_reference(model::JuMP.Model, k::Int)
    nx = length(model[:μₜ])
    costates = model[:μₜ₊₁]
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

function solve!(sddp::DualSDDP, model::JuMP.Model, μₜ::Vector{Float64})
    fix.(model[:μₜ], μₜ)
    optimize!(model)
    v = JuMP.all_variables(model)
    @assert termination_status(model) ∈ sddp.valid_statuses println(termination_status(model))
    return
end

function next!(
    sddp::DualSDDP,
    model::JuMP.Model,
    μₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
)
    solve!(sddp, model, μₜ)
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
    solve!(sddp, model, μₜ)
    x = dual.(FixRef.(model[:μₜ]))
    γ = objective_value(model) - dot(x, μₜ)
    add_cut!(Dₜ, x, γ)
    return x
end

function synchronize!(::DualSDDP, model::JuMP.Model, Dₜ₊₁::PolyhedralFunction)
    nw = length(model[:θ])
    for k in 1:nw
        μk = _next_costate_reference(model, k)
        @constraint(model, model[:θ][k] >= Dₜ₊₁.λ[end, :]' * μk + Dₜ₊₁.γ[end])
    end
    return
end

#=
    Algorithm
=#

function fenchel_transform(solver::DualSDDP, D::PolyhedralFunction, x)
    nx = dimension(D)
    model = Model()
    @variable(model, solver.lipschitz_lb <= λ[1:nx] <= 0.0)
    @variable(model, θ)
    for (xk, βk) in eachcut(D)
        @constraint(model, θ >= dot(xk, λ) + βk)
    end
    @objective(model, Max, dot(x, λ) - θ)

    JuMP.set_optimizer(model, solver.optimizer)
    JuMP.optimize!(model)

    return JuMP.objective_value(model), JuMP.value.(λ)
end

function solve!(
    solver::DualSDDP,
    hdm::HazardDecisionModel,
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int = 1,
)
    (verbose > 0) && println("** Minicut SDDP **")

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
            # TODO: clean reformulation
            obj_expr = objective_function(model)
            @objective(model, Min, -obj_expr)
        end
        JuMP.set_optimizer(model, solver.optimizer)
    end
    # Run
    for i in 1:n_iter
        lb, p₀ = fenchel_transform(solver, D[1], x₀)
        scenario = sample(Ξ)
        dual_trajectory = forward_pass(solver, hdm, models, scenario, p₀)
        primal_trajectory = backward_pass!(solver, hdm, models, dual_trajectory, D)
        lb, p₀ = fenchel_transform(solver, D[1], x₀)
        if (verbose > 0) && (mod(i, verbose) == 0)
            @printf(" %4i %15.6e\n", i, lb)
        end
    end
    return models
end

