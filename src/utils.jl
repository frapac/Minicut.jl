
abstract type AbstractRandomVariable end

struct DiscreteRandomVariable{T}
    weights::Vector{T}
    supports::Array{T,2}
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

function Base.rand(v::DiscreteRandomVariable)
    n = findfirst(cumsum(v.weights) .>= rand())
    return v.supports[:, n]
end

function find_outcome(v::DiscreteRandomVariable, x::Vector)
    idx = -1
    for k in 1:length(v)
        vk = view(v.supports, :, k)
        if isequal(vk, x)
            idx = k
            break
        end
    end
    return idx
end

function sample!(vs::Vector{DiscreteRandomVariable{T}}, scenario::Matrix{T}) where {T}
    for (t, v) in enumerate(vs)
        scenario[:, t] .= rand(v)
    end
    return scenario
end

function sample(vs::Vector{DiscreteRandomVariable{T}}) where {T}
    # All v in vs should share the same dimension
    m = dimension(vs[1])
    scenario = zeros(T, m, length(vs))
    return sample!(vs, scenario)
end

function sample(vs::Vector{DiscreteRandomVariable{T}}, n_scenarios) where {T}
    return [sample(vs) for _ in 1:n_scenarios]
end

function number_nodes(vs::Vector{DiscreteRandomVariable{T}}) where {T}
    n = length(vs)
    counts = zeros(Int, n + 1)
    counts[1] = 1
    for t in 1:n
        counts[t+1] = counts[t] * length(vs[t])
    end
    return counts
end

function header()
    return println("This is Minicut 0.1.0.\n")
end
