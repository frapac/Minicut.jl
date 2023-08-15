
abstract type AbstractSDDP <: AbstractStochasticOptimizer end

include("primal_sddp.jl")
include("dual_sddp.jl")
include("mixed_sddp.jl")
include("regularized_sddp.jl")

#=
    Generic Forward pass
=#

function forward_pass!(
    sddp::AbstractSDDP,
    tree::AbstractMultiStageModel,
    scenarios::InSampleScenario{Float64},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
)
    Ξ = uncertainties(tree.model)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    for stage in tree.stages
        wₜ = scenarios.values[:, stage.t]
        xₜ = next!(sddp, stage, xₜ, Ξ[stage.t], wₜ)
        trajectory[:, stage.t+1] .= xₜ
    end
    return trajectory
end

function forward_pass(
    sddp::AbstractSDDP,
    tree::AbstractMultiStageModel,
    scenario::AbstractScenario,
    initial_state::Vector{Float64},
)
    T = horizon(tree.model)
    primal_trajectory = fill(0.0, length(initial_state), T + 1)
    forward_pass!(sddp, tree, scenario, initial_state, primal_trajectory)
    return primal_trajectory
end

function forward_pass(
    sddp::AbstractSDDP,
    tree::AbstractMultiStageModel,
    scenarios::Vector{S},
    initial_state::Vector{Float64},
) where S <: AbstractScenario
    trajectories = Matrix{Float64}[]
    for scenario in scenarios
        push!(trajectories, forward_pass(sddp, tree, scenario, initial_state))
    end
    return trajectories
end

#=
    Backward pass
=#

function backward_pass!(
    sddp::AbstractSDDP,
    tree::AbstractMultiStageModel,
    primal_trajectory::Array{Float64,2},
    V::Vector{PolyhedralFunction},
)
    T = horizon(tree.model)
    @assert length(V) == T
    Ξ = uncertainties(tree.model)
    nscen = length(primal_trajectory)
    trajectory = zeros(size(primal_trajectory))
    stage = final_stage(tree)
    while !isnothing(stage)
        t = stage.t
        trajectory[:, t] .= previous!(sddp, stage, primal_trajectory[:, t], Ξ[t], V[t])
        stage = stage.parent
    end
    return trajectory
end

function backward_pass!(
    sddp::AbstractSDDP,
    tree::AbstractMultiStageModel,
    primal_trajectory::Vector{Array{Float64, 2}},
    V::Vector{PolyhedralFunction},
)
    T, nx = horizon(tree.model), number_states(tree.model)
    nscen = length(primal_trajectory)
    @assert length(V) == T
    Ξ = uncertainties(tree.model)
    dual_trajectory = fill(zeros(nx, T), nscen)
    stage = final_stage(tree)
    while !isnothing(stage)
        t = stage.t
        for s in 1:nscen
            x = primal_trajectory[s][:, t]
            λ = previous!(sddp, stage, x, Ξ[t], V[t])
            dual_trajectory[s][:, t] .= λ
        end
        stage = stage.parent
    end
    return dual_trajectory
end

#=
    Simulation
=#

function simulate!(
    sddp::AbstractSDDP,
    tree::AbstractMultiStageModel,
    initial_state::Vector{Float64},
    scenarios::Vector{InSampleScenario{Float64}},
)
    Ξ = uncertainties(tree.model)
    n_scenarios = length(scenarios)
    n_states = number_states(tree.model)
    xₜ = zeros(n_states)
    costs = zeros(n_scenarios)
    for k in 1:n_scenarios
        xₜ .= initial_state
        for stage in tree.stages
            t = stage.t
            ξ = scenarios[k].values[:, t]
            xₜ = next!(sddp, stage, xₜ, Ξ[t], ξ)
            costs[k] += stage_objective_value(sddp, stage, tree.model, t)
        end
    end
    return costs
end

