
abstract type AbstractRandomVariable end

struct DiscreteRandomVariable{T}
    weights::Vector{T}
    supports::Array{T, 2}
end

function DiscreteRandomVariable(π, supports)
    @assert sum(π) == 1.0
    return DiscreteRandomVariable(π, supports)
end

Base.length(v::DiscreteRandomVariable) = size(v.supports, 2)
dimension(v::DiscreteRandomVariable) = size(v.supports, 1)

function Base.getindex(v::DiscreteRandomVariable, n::Int)
    @assert 1 <= n <= size(v.supports, 2)
    return (weights[n], view(supports[:, n]))
end

function sample(v::DiscreteRandomVariable)
    n = findfirst(cumsum(v.weights) .>= rand())
    w = v.supports[:, n]
    return (n, w)
end

function Base.rand(v::DiscreteRandomVariable)
    return sample(v)[2]
end

function find_outcome(v::DiscreteRandomVariable, x::Vector)
    idx = -1
    for k in 1:length(v)
        vk = view(v.supports, :, k)
        if isequal(vk, x)
            idx = k ; break
        end
    end
    return idx
end

abstract type AbstractScenario end

struct InSampleScenario{T} <: AbstractScenario
    path::Vector{Int}
    values::Matrix{T}
end

function InSampleScenario(T::Int, m::Int)
    return InSampleScenario{Float64}(
        zeros(Int, T),
        zeros(Float64, m, T)
    )
end

function sample!(vs::Vector{DiscreteRandomVariable{T}}, scen::InSampleScenario{T}) where T
    for (t, v) in enumerate(vs)
        (k, w) = sample(v)
        scen.path[t] = k
        scen.values[:, t] .= w
    end
    return scen
end

function sample(vs::Vector{DiscreteRandomVariable{T}}) where T
    # All v in vs should share the same dimension
    scenario = InSampleScenario(length(vs), dimension(vs[1]))
    return sample!(vs, scenario)
end

function sample(vs::Vector{DiscreteRandomVariable{T}}, n_scenarios) where T
    return [sample(vs) for _ in 1:n_scenarios]
end

function number_nodes(vs::Vector{DiscreteRandomVariable{T}}) where T
    n = length(vs)
    counts = zeros(Int, n+1)
    counts[1] = 1
    for t in 1:n
        counts[t+1] = counts[t] * length(vs[t])
    end
    return counts
end

function header()
    println("This is Minicut 0.1.0.\n")
end

function init_save(hdm::HazardDecisionModel, n_iter::Int)
    data = DataFrame()
    data[!, :horizon] = [horizon(hdm)]
    data[!, :n_x] = [number_states(hdm)]
    data[!, :n_iter] = [n_iter]
    data[!, :n_xi] = [length(uncertainties(hdm))]

    # Initializing the DataFrames to be filled during the run
    timers = DataFrame()
    timers[!, :iteration] = 1:n_iter
    timers[!, :time_primal_forward] .= 0.0
    timers[!, :time_primal_backward] .= 0.0
    timers[!, :time_dual_forward] .= 0.0
    timers[!, :time_dual_backward] .= 0.0

    lb = DataFrame(-Inf*ones(Float64, (n_iter, horizon(hdm))), :auto)
    insertcols!(lb, 1, :iteration => 1:n_iter)

    ub = DataFrame(Inf*ones(Float64, (n_iter, horizon(hdm))), :auto)
    insertcols!(ub, 1, :iteration => 1:n_iter)

    return (data=data, timers=timers, lb=lb, ub=ub)
end

