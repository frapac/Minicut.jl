#=
    SDDP
=#

struct SDDP <: AbstractSDDP
    optimizer::Any
    valid_statuses::Vector{MOI.TerminationStatusCode}
end

SDDP(optimizer) = SDDP(optimizer, [MOI.OPTIMAL])

introduce(::SDDP) = "Primal SDDP"

#=
    One-stage problem
=#

function initialize!(::SDDP, model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
    @variable(model, θ)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(model, θ >= λ' * model[_CURRENT_STATE] + γ)
    end
    obj_expr = objective_function(model)
    @objective(model, Min, obj_expr + θ)
    return
end

function solve_stage_problem!(sddp::SDDP, model::JuMP.Model, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64})
    fix.(model[_PREVIOUS_STATE], xₜ, force = true)
    fix.(model[_UNCERTAINTIES], ξₜ₊₁, force = true)
    optimize!(model)
    if termination_status(model) ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve primal subproblem: solver's return status is $(termination_status(model))")
    end
    return
end

fetch_cut(sddp::SDDP, model::JuMP.Model) = dual.(FixRef.(model[_PREVIOUS_STATE]))

function stage_objective_value(sddp::SDDP, model::JuMP.Model, hdm::HazardDecisionModel, t)
    if t == horizon(hdm)
        return JuMP.objective_value(model)
    else
        Vx = JuMP.value.(model[_VALUE_FUNCTION])
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
    solve_stage_problem!(sddp, model, xₜ, ξₜ₊₁)
    return value.(model[_CURRENT_STATE])
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
        solve_stage_problem!(sddp, model, xₜ, ξₜ₊₁[:, i])
        λᵢ = fetch_cut(sddp, model)
        axpy!(πᵢ, λᵢ, λ)
        γ += πᵢ * (objective_value(model) - dot(λᵢ, xₜ))
    end
    add_cut!(Vₜ, λ, γ)
    return λ
end

function synchronize!(::SDDP, model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
    @constraint(model, model[_VALUE_FUNCTION] >= Vₜ₊₁.λ[end, :]' * model[_CURRENT_STATE] + Vₜ₊₁.γ[end])
    return
end

function build_stage_models(solver::SDDP, hdm::HazardDecisionModel, V::Vector{PolyhedralFunction})
    T = horizon(hdm)
    models = [stage_model(hdm, t) for t in 1:T]
    for (t, model) in enumerate(models)
        if t < T
            initialize!(solver, model, V[t+1])
        end
        JuMP.set_optimizer(model, solver.optimizer)
    end
    return models
end

function solve!(
    solver::SDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int = 1,
)
    (verbose > 0) && header()

    models = build_stage_models(solver, hdm, V)
    Ξ = uncertainties(hdm)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        println("    Solver:  ", solver.optimizer.optimizer_constructor)
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s\n", "-"^4, "-"^15)
        @printf(" %4s %15s\n", "#it", "LB")
    end

    tic = time()
    # Run
    for i in 1:n_iter
        scen = sample(Ξ)
        primal_trajectory = forward_pass(solver, hdm, models, scen, x₀)
        dual_trajectory = backward_pass!(solver, hdm, models, primal_trajectory, V)
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            @printf(" %4i %15.6e\n", i, lb)
        end
    end

    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s\n\n", "-"^4, "-"^15)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
    end

    return models
end

# Helper function
function sddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    seed=0,
    n_iter=500,
    verbose::Int = 1,
    lower_bound=-1e6,
    valid_statuses=[MOI.OPTIMAL],
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    V = [PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
    solver = SDDP(optimizer, valid_statuses)
    models = solve!(solver, hdm, V, x₀; n_iter=n_iter, verbose=verbose)
    return (cuts=V, models=models, lower_bound=V[1](x₀))
end

