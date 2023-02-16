
abstract type AbstractSDDP <: AbstractStochasticOptimizer end

include("primal_sddp.jl")
include("dual_sddp.jl")

#=
    Generic Forward pass
=#

function forward_pass!(
    sddp::AbstractSDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    uncertainty_scenario::Array{Float64, 2},
    initial_state::Vector{Float64},
    trajectory::Array{Float64, 2},
)
    Ξ = uncertainties(hdm)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    for (t, ξₜ₊₁) in enumerate(eachcol(uncertainty_scenario))
        xₜ = next!(sddp, models[t], xₜ, Ξ[t], collect(ξₜ₊₁))
        trajectory[:, t+1] .= xₜ
    end
    return trajectory
end

function forward_pass(
    sddp::AbstractSDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    uncertainty_scenario::Array{Float64, 2},
    initial_state::Vector{Float64},
)
    horizon = size(uncertainty_scenario, 2)
    primal_trajectory = fill(0.0, length(initial_state), horizon + 1)
    return forward_pass!(sddp, hdm, models, uncertainty_scenario, initial_state, primal_trajectory)
end

#=
    Backward pass
=#

function backward_pass!(
    sddp::AbstractSDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    primal_trajectory::Array{Float64,2},
    V::Vector{PolyhedralFunction},
)
    T = length(models)
    Ξ = uncertainties(hdm)
    @assert length(V) == T
    trajectory = zeros(size(primal_trajectory))
    # Final time
    trajectory[:, T] .= previous!(sddp, models[T], primal_trajectory[:, T], Ξ[T], V[T])
    # Reverse pass
    @inbounds for t in reverse(1:T-1)
        synchronize!(sddp, models[t], V[t+1])
        trajectory[:, t] .= previous!(sddp, models[t], primal_trajectory[:, t], Ξ[t], V[t])
    end
    return trajectory
end

#=
    Simulation
=#

function simulate!(
    sddp::AbstractSDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    initial_state::Vector{Float64},
    uncertainty_scenario::Vector{Array{Float64, 2}},
)
    Ξ = uncertainties(hdm)
    n_scenarios = length(uncertainty_scenario)
    n_states = number_states(hdm)
    xₜ = zeros(n_states)
    costs = zeros(n_scenarios)
    for k in 1:n_scenarios
        xₜ .= initial_state
        for t in 1:horizon(hdm)
            ξ = uncertainty_scenario[k][:, t]
            xₜ = next!(sddp, models[t], xₜ, Ξ[t], ξ)
            costs[k] += stage_objective_value(sddp, models[t], hdm, t)
        end
    end
    return costs
end

function sample_trajectory!(
    sddp::AbstractSDDP,
    hdm::HazardDecisionModel,
    models::Vector{JuMP.Model},
    initial_state::Vector{Float64},
    uncertainty_scenario::Vector{Array{Float64, 2}},
)
    Ξ = uncertainties(hdm)
    n_states, n_scenarios = number_states(hdm), length(uncertainty_scenario)
    trajectory = zeros(n_scenarios, horizon(hdm)+1, n_states)
    for k in 1:n_scenarios
        trajectory[k, 1, :] .= initial_state
    end
    # NB: inverting the for loops is more efficient
    for t in 1:horizon(hdm)
        for k in 1:n_scenarios
            ξ = uncertainty_scenario[k][:, t]
            trajectory[k, t+1, :] .= next!(sddp, models[t], trajectory[k, t, :], Ξ[t], ξ)
        end
    end
    return trajectory
end

