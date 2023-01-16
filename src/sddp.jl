
struct SDDP <: AbstractStochasticOptimizer
    optimizer::Any
end

#=
    One-stage problem
=#

function initialize!(model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
    @variable(model, θ)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(model, θ >= λ' * model[:xₜ₊₁] + γ)
    end
    obj_expr = objective_function(model)
    @objective(model, Min, obj_expr + θ)
    return
end

function solve!(::HazardDecisionModel, model::JuMP.Model, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64})
    fix.(model[:xₜ], xₜ, force = true)
    fix.(model[:ξₜ₊₁], ξₜ₊₁, force = true)
    optimize!(model)
    @assert termination_status(model) ∈ [MOI.OPTIMAL, MOI.OTHER_ERROR]
    return
end

function stage_objective_value(hdm::HazardDecisionModel, model::JuMP.Model, t)
    Vx = JuMP.value.(model[:θ])
    if t == horizon(hdm)
        return JuMP.objective_value(model)
    end
    return JuMP.objective_value(model) - Vx
end

function next!(hdm::HazardDecisionModel, model::JuMP.Model, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64})
    solve!(hdm, model, xₜ, ξₜ₊₁)
    return (value.(model[:uₜ₊₁]), value.(model[:xₜ₊₁]))
end

function previous!(hdm::HazardDecisionModel, model::JuMP.Model, Vₜ::PolyhedralFunction, t::Int, xₜ::Vector{Float64})
    nx = length(xₜ)
    ξ = uncertainties(hdm)[t]
    πₜ₊₁, ξₜ₊₁ = ξ.weights, ξ.supports
    λ = zeros(nx)
    γ = 0.0
    for (i, πᵢ) in enumerate(πₜ₊₁)
        solve!(hdm, model, xₜ, ξₜ₊₁[:, i])
        λᵢ = dual.(FixRef.(model[:xₜ]))
        axpy!(πᵢ, λᵢ, λ)
        γ += πᵢ * (objective_value(model) - dot(λᵢ, xₜ))
    end
    add_cut!(Vₜ, λ, γ)
    return λ
end

function synchronize!(::HazardDecisionModel, model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
    @constraint(model, model[:θ] >= Vₜ₊₁.λ[end, :]' * model[:xₜ₊₁] + Vₜ₊₁.γ[end])
    return
end

#=
    Forward pass
=#

function forward_pass!(
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    ξs::Array{Float64, 2},
    x₀::Vector{Float64},
    primal_trajectory::Array{Float64, 2},
)
    xₜ = copy(x₀)
    primal_trajectory[:, 1] .= x₀
    for (t, ξₜ₊₁) in enumerate(eachcol(ξs))
        uₜ₊₁, xₜ = next!(hdm, models[t], xₜ, collect(ξₜ₊₁))
        primal_trajectory[:, t+1] .= xₜ
    end
    return primal_trajectory
end

function forward_pass(
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    ξs::Array{Float64, 2},
    x₀::Vector{Float64},
)
    T = size(ξs, 2)
    primal_trajectory = fill(0.0, length(x₀), T + 1)
    return forward_pass!(hdm, models, ξs, x₀, primal_trajectory)
end

#=
    Backward pass
=#

function backward_pass!(
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    primal_trajectory::Array{Float64,2},
    V::Vector{PolyhedralFunction},
)
    T = length(models)
    @assert length(V) == T + 1
    dual_trajectory = zeros(size(primal_trajectory))
    # Final time
    dual_trajectory[:, T] .= previous!(hdm, models[T], V[T], T, primal_trajectory[:, T])
    # Reverse pass
    @inbounds for t in reverse(1:T-1)
        synchronize!(hdm, models[t], V[t+1])
        dual_trajectory[:, t] .= previous!(hdm, models[t], V[t], t, primal_trajectory[:, t])
    end
    return dual_trajectory
end

#=
    Simulation
=#

function simulate!(
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    x₀::Vector{Float64},
    ξs::Vector{Array{Float64, 2}},
)
    n_scenarios = length(ξs)
    n_states = number_states(hdm)
    T = horizon(hdm)
    xₜ = zeros(n_states)
    costs = zeros(n_scenarios)
    for k in 1:n_scenarios
        xₜ .= x₀
        for t in 1:T
            ξ = ξs[k][:, t]
            xₜ .= next!(hdm, models[t], xₜ, ξ)[2]
            costs[k] += stage_objective_value(hdm, models[t], t)
        end
    end
    return costs
end

#=
    SDDP
=#

function solve!(
    solver::SDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int = 1,
)
    (verbose > 0) && println("** Minicut SDDP **")

    T = horizon(hdm)
    # Initialize
    models = [stage_model(hdm, t) for t in 1:T]
    for (t, Vₜ₊₁) in enumerate(V[2:end])
        initialize!(models[t], Vₜ₊₁)
        JuMP.set_optimizer(models[t], solver.optimizer)
    end

    Ξ = uncertainties(hdm)

    # Run
    for i in 1:n_iter
        scen = sample(Ξ)
        primal_trajectory = forward_pass(hdm, models, scen, x₀)
        backward_pass!(hdm, models, primal_trajectory, V)
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            @printf(" %4i %15.6e\n", i, lb)
        end
    end
    return models
end

