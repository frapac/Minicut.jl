#=
    Regularized primal SDDP
=#

struct RegularizedPrimalSDDP <: AbstractSDDP
    primal_sddp::SDDP
    dual_sddp::DualSDDP
    tau::Float64
    mixing::Float64
    name::String
end 

introduce(regsddp::RegularizedPrimalSDDP) = regsddp.name

function solve_stage_problem!(sddp::SDDP, model::JuMP.Model, V::Vector{PolyhedralFunction}, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64}, ℓ::Float64, τ::Float64, t, T)
    fix.(model[_PREVIOUS_STATE], xₜ, force=true)
    fix.(model[_UNCERTAINTIES], ξₜ₊₁, force=true)
    cost = JuMP.objective_function(model)

    if t < T
        @variable(model, θ)
        for (λ, γ) in eachcut(V[t+1])
            @constraint(model, θ >= λ' * model[_CURRENT_STATE] + γ)
        end
        @constraint(model, θ + cost >= ℓ) 
        @objective(model, Min, θ + cost + (1 / (2 * τ)) * sum(vcat(model[_CURRENT_STATE], model[_CURRENT_CONTROL]) .^ 2))
    else
        @objective(model, Min, cost)
    end

    optimize!(model)
    if termination_status(model) ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve primal regularized subproblem: solver's return status is $(termination_status(model))")
    end
    res = JuMP.objective_value(model)
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
    ξₜ₊₁::Vector{Float64},
)
    set_optimizer(model, sddp.optimizer)
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
    fix.(model[_PREVIOUS_STATE], xₜ, force=true)
    fix.(model[_UNCERTAINTIES], ξₜ₊₁, force=true)
    JuMP.set_optimizer(model, dual_sddp.optimizer)
    # future states
    xf = model[_CURRENT_STATE]
    # number of cuts
    n_cuts = ncuts(D)
    # Lipschitz constant
    lipschitz = dual_sddp.lipschitz_ub

    # define simplex Λ
    @variable(model, sigma[1:n_cuts] >= 0.0)
    @variable(model, xabs[1:size(xf)[1]])
    @variable(model, y[1:size(xf)[1]])

    @constraint(model, sum(sigma) == 1.0)
    # we build the inner approximation all in once
    @constraint(model, xabs .>= xf - y) #norm1
    @constraint(model, xabs .>= y - xf)
    @constraint(model, y .== sum(sigma[i] * D.λ[i, :] for i in 1:ncuts(D)))

    cost_fct = objective_function(model)
    @objective(model, Min, cost_fct - sum(sigma[i] * D.γ[i] for i in 1:n_cuts) + lipschitz * sum(xabs))
    optimize!(model)
    return objective_value(model)
end

function next!(
    Regsddp::RegularizedPrimalSDDP,
    model::JuMP.Model,
    V::Vector{PolyhedralFunction},
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64},
    ℓ::Float64,
    τ::Float64,
    t,
    T,
)
    JuMP.set_optimizer(model, Regsddp.primal_sddp.optimizer)
    solve_stage_problem!(Regsddp.primal_sddp, model, V, xₜ, ξₜ₊₁, ℓ, τ, t, T)
    return value.(model[_CURRENT_STATE])
end

function reg_forward_pass!(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    dual_models::Vector{JuMP.Model},
    V::Vector{PolyhedralFunction},
    D::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
    τ::Float64,
    upperbounds::Array{Float64}
)
    Ξ = uncertainties(hdm)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    for (t, ξₜ₊₁) in enumerate(eachcol(uncertainty_scenario))
        xi = collect(ξₜ₊₁)
        # Lower-bound.
        lb = lowerbound(Regsddp.primal_sddp, primal_models[t], xₜ, xi)
        # Upper-bound.
        ubmodel = stage_model(hdm, t)
        if t < horizon(hdm)
            ub = upperbound(Regsddp.dual_sddp, ubmodel, xₜ, xi, D[t+1])
        else
            ub = lb
        end
        upperbounds[t] = ub
        # Regularization level ; Adaptative combination between lb and ub depending on the relative gap
        relative_gap = abs((ub - lb)/lb)
        if relative_gap > 0.1
            mix_ublb = min(relative_gap, 1)
        else # When low enough, both ub and lb are equally good approximations
            mix_ublb = 0.5/t
        end
        ℓ = mix_ublb * lb + (1.0 - mix_ublb) * ub
        reg_model = stage_model(hdm, t)
        xₜ = next!(Regsddp, reg_model, V, xₜ, xi, ℓ, τ, t, horizon(hdm))
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
    D::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    τ::Float64
)
    horizon = size(uncertainty_scenario, 2)
    primal_trajectory = fill(0.0, length(initial_state), horizon + 1)
    upperbounds = zeros(Float64, horizon)
    return reg_forward_pass!(Regsddp, hdm, primal_models, dual_models, V, D, uncertainty_scenario, initial_state, primal_trajectory, τ, upperbounds), upperbounds
end

function solve!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    n_warming = 10,
    verbose::Int=1,
    τ=1e8,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)
    dual_models = build_stage_models(solver.dual_sddp, hdm, D)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        println("Warm-up: $(n_warming) iterations")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Warming up
    ub = Inf
    tic = time()
    for i in 1:n_warming
        scenario = sample(Ξ)
        # Primal
        primal_trajectory = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
        dual_trajectory = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectory, V)
        # Dual            trajectory[k, t+1, :] .= next!(sddp, models[t], trajectory[k, t, :], Ξ[t], ξ)

        backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectory, D)
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)

        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
    end
    if verbose > 0
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Run
    #ub = Inf
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    for i in 1:n_iter
        scenario = sample(Ξ)
        # Primal
        primal_trajectory, ub_tmp = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
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
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    n_warming = 10,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing, "Regularized Primal SDDP")
    primal_models, dual_models = solve!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, n_warming=n_warming, verbose=verbose, τ=τ)

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

