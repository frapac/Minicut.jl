#=
    SDDP
=#

struct SDDP <: AbstractSDDP
    optimizer::Any
    valid_statuses::Vector{MOI.TerminationStatusCode}
end

SDDP(optimizer) = SDDP(optimizer, [MOI.OPTIMAL])

#=
    One-stage problem
=#

function initialize!(::SDDP, model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
    @variable(model, θ)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(model, θ >= λ' * model[:xₜ₊₁] + γ)
    end
    obj_expr = objective_function(model)
    @objective(model, Min, obj_expr + θ)
    return
end

function solve!(sddp::SDDP, model::JuMP.Model, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64})
    fix.(model[:xₜ], xₜ, force = true)
    fix.(model[:ξₜ₊₁], ξₜ₊₁, force = true)
    optimize!(model)
    @assert termination_status(model) ∈ sddp.valid_statuses
    return
end

function stage_objective_value(sddp::SDDP, model::JuMP.Model, hdm::HazardDecisionModel, t)
    if t == horizon(hdm)
        return JuMP.objective_value(model)
    else
        Vx = JuMP.value.(model[:θ])
        return JuMP.objective_value(model) - Vx
    end
end

function next!(
    sddp::SDDP,
    model::JuMP.Model,
    xₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
)
    solve!(sddp, model, xₜ, ξₜ₊₁)
    return value.(model[:xₜ₊₁])
end

function previous!(
    sddp::SDDP,
    model::JuMP.Model,
    xₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    Vₜ::PolyhedralFunction,
)
    nx = length(xₜ)
    πₜ₊₁, ξₜ₊₁ = ξ.weights, ξ.supports
    λ = zeros(nx)
    γ = 0.0
    for (i, πᵢ) in enumerate(πₜ₊₁)
        solve!(sddp, model, xₜ, ξₜ₊₁[:, i])
        λᵢ = dual.(FixRef.(model[:xₜ]))
        axpy!(πᵢ, λᵢ, λ)
        γ += πᵢ * (objective_value(model) - dot(λᵢ, xₜ))
    end
    add_cut!(Vₜ, λ, γ)
    return λ
end

function synchronize!(::SDDP, model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
    @constraint(model, model[:θ] >= Vₜ₊₁.λ[end, :]' * model[:xₜ₊₁] + Vₜ₊₁.γ[end])
    return
end

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
    for (t, model) in enumerate(models)
        if t < T
            initialize!(solver, model, V[t+1])
        end
        JuMP.set_optimizer(model, solver.optimizer)
    end

    Ξ = uncertainties(hdm)

    # Run
    for i in 1:n_iter
        scen = sample(Ξ)
        primal_trajectory = forward_pass(solver, hdm, models, scen, x₀)
        backward_pass!(solver, hdm, models, primal_trajectory, V)
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            @printf(" %4i %15.6e\n", i, lb)
        end
    end
    return models
end

