#=
    Regularized primal SDDP
=#

struct RegularizedPrimalSDDP <: AbstractSDDP
    primal_sddp::SDDP
    dual_sddp::DualSDDP
    tau::Float64
    mixing::Float64
end

introduce(::RegularizedPrimalSDDP) = "Regularized Primal SDDP"

# function initialize!(::SDDP, model::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
#     @variable(model, θ)
#     for (λ, γ) in eachcut(Vₜ₊₁)
#         @constraint(model, θ >= λ' * model[_CURRENT_STATE] + γ)
#     end
#     obj_expr = objective_function(model)
#     @objective(model, Min, obj_expr + θ)
#     return
# end

# Should change later: currently it re adds a ρ variable everytime
function solve_stage_problem!(sddp::SDDP, model::JuMP.Model, V::Vector{PolyhedralFunction}, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64}, ℓ::Float64, τ::Float64, t, T)
    fix.(model[_PREVIOUS_STATE], xₜ, force=true)
    fix.(model[_UNCERTAINTIES], ξₜ₊₁, force=true)
    cost = JuMP.objective_function(model)

    @variable(model, θ)
    if 0 < t < T
        for (λ, γ) in eachcut(V[t+1])
            @constraint(model, θ >= λ' * model[_CURRENT_STATE] + γ + cost)
        end
    else
        @constraint(model, θ >= cost)
    end
    @constraint(model, ℓ <= θ)

    @objective(model, Min, θ + (1 / (2 * τ)) * sum(vcat(model[_CURRENT_STATE], model[_CURRENT_CONTROL]) .^ 2))
    optimize!(model)
    if termination_status(model) ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve primal regularized subproblem: solver's return status is $(termination_status(model))")
    end
    return
end

function stage_cost(sddp::SDDP, model::JuMP.Model, t, T)
    if t == T
        return objective_function(model)
    else
        return objective_function(model) - model[_VALUE_FUNCTION]
    end
end

function lowerbound(
    sddp::SDDP,
    model::JuMP.Model,
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64}
)
    solve_stage_problem!(sddp, model, xₜ, ξₜ₊₁)
    return objective_value(model)
end

function upperbound(
    dual_sddp::DualSDDP,
    model::JuMP.Model,
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64},
    D::PolyhedralFunction,
)
    # future states
    xf = model[:xₜ₊₁]
    # number of cuts
    n_cuts = V.ncuts()
    # Lipschitz constant
    lipschitz = V.lipschitz_constant()

    # define simplex Λ
    @variable(model, eta[1:n_cuts] >= 0.0)
    @variable(model, no1[1:size(xf)[1]])
    @variable(model, x_alt[1:size(xf)[1]])
    @constraint(model, sum(eta) == 1.0)

    # we build the inner approximation all in once
    @constraint(model, no1 .>= xf - x_alt) #norm1
    @constraint(model, no1 .>= x_alt - xf)
    @constraint(model, x_alt .== sum(eta[i] * D.λ[i, :] for i in 1:n_cuts))

    cost_fct = objective_function(model)
    @objective(model, Min, cost_fct - sum(eta[i] * D.γ[i] for i in 1:n_cuts) + lipschitz * sum(no1))
    optimize!(model)
    return objective_value(model)
end


function next!(
    Regsddp::RegularizedPrimalSDDP,
    model::JuMP.Model,
    V::Vector{PolyhedralFunction},
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64},
    lb::Float64,
    τ::Float64,
    t,
    T,
)
    JuMP.set_optimizer(model, Regsddp.primal_sddp.optimizer)
    solve_stage_problem!(Regsddp.primal_sddp, model, V, xₜ, ξₜ₊₁, lb, τ, t, T)
    return value.(model[_CURRENT_STATE])
end

function reg_forward_pass!(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    dual_models::Vector{JuMP.Model},
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
    τ::Float64
)
    Ξ = uncertainties(hdm)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    for (t, ξₜ₊₁) in enumerate(eachcol(uncertainty_scenario))
        xi = collect(ξₜ₊₁)
        lb = lowerbound(Regsddp.primal_sddp, primal_models[t], xₜ, xi)
        model = stage_model(hdm, t)
        xₜ = next!(Regsddp, model, V, xₜ, xi, lb, τ, t, horizon(hdm))
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end

function reg_forward_pass(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    dual_models::Vector{JuMP.Model},
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    τ::Float64
)
    horizon = size(uncertainty_scenario, 2)
    primal_trajectory = fill(0.0, length(initial_state), horizon + 1)
    return reg_forward_pass!(Regsddp, hdm, primal_models, dual_models, V, uncertainty_scenario, initial_state, primal_trajectory, τ)
end

function solve!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int=1,
    τ=1e8
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)
    dual_models = build_stage_models(solver.dual_sddp, hdm, D)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        println("    Primal solver....:  ", solver.primal_sddp.optimizer.optimizer_constructor)
        println("    Dual solver......:  ", solver.dual_sddp.optimizer.optimizer_constructor)
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Run
    #ub = Inf
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    tic = time()
    for i in 1:n_iter
        scenario = sample(Ξ)
        # Primal
        primal_trajectory = reg_forward_pass(solver, hdm, primal_models, dual_models, V, scenario, x₀, τ)
        backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectory, V)
        # Dual
        dual_trajectory = forward_pass(solver.dual_sddp, hdm, dual_models, scenario, p₀)
        backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectory, D)
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)

        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
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
function regularizedsddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    mixing=0.2,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL]
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Polyhedral functions.
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    # Solvers.
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing)
    primal_models, dual_models = solve!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, verbose=verbose, τ=τ)

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

