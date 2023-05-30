#=
    Regularized SDDP with the Minimum Upper Bound rule to compute the upperbounds [van Ackooij et al (2019)]
=#

function minub(
    sddp::SDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    n_scenarios::Int,
    initial_state::Vector{Float64},
    Ξ::Vector{DiscreteRandomVariable{Float64}},
    t::Int
)
    z = zeros(Float64, length(Ξ[t]))
    for j in 1:length(Ξ[t])
        x_j = next!(sddp, models[t], initial_state, Ξ[t], ξ)
        z[j] = montecarlo(sddp, hdm, models[t+1:horizon(hdm)], n_scenarios, x_j, Ξ[t+1:T], t)
    end
    return minimum(z)
end

function minub_forward_pass!(
    Regsddp::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    primal_models::Vector{JuMP.Model},
    V::Vector{PolyhedralFunction},
    uncertainty_scenario::Array{Float64,2},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
    τ::Float64
)
    Ξ = uncertainties(hdm)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    cum_cost = 0.0

    for (t, ξₜ₊₁) in enumerate(eachcol(uncertainty_scenario))
        xi = collect(ξₜ₊₁)
        # Lower-bound.
        lb = lowerbound(Regsddp.primal_sddp, primal_models[t], xₜ, xi)

        # Upper-bound.
        ubmodel = stage_model(hdm, t)
        
        if t < horizon(hdm)
            ub = init_ub - cum_cost
        else
            ub = lb
        end
        # Regularization level ; Adaptative combination between lb and ub depending on the relative gap
        relative_gap = abs((ub - lb)/lb)
        mixing = min(0.1*relative_gap, 1) # If gap is too big, favor the lb from classic SDDP
        #mixing = .5
        ℓ = mixing * lb + (1.0 - mixing) * ub
        
        model = stage_model(hdm, t)
        xₜ = next!(Regsddp, model, V, xₜ, xi, ℓ, τ, t, horizon(hdm))
        cum_cost += stage_objective_value(Regsddp.primal_sddp, model, hdm, t)
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end