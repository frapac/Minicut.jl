mutable struct PolyhedralFunction
    λ::Array{Float64,2}
    γ::Array{Float64,1}
end

ncuts(V::PolyhedralFunction) = size(V.λ, 1)
dimension(V::PolyhedralFunction) = size(V.λ, 2)
eachcut(V::PolyhedralFunction) = zip(eachrow(V.λ), V.γ)

PolyhedralFunction(nx::Int) = PolyhedralFunction(zeros(0, nx), Float64[])
PolyhedralFunction(nx::Int, lb::Float64) = PolyhedralFunction(zeros(1, nx), [lb])

function lipschitz_constant(V::PolyhedralFunction, pnorm::Real=1)
    return maximum([norm(λ, pnorm) for λ in eachrow(V.λ)])
end

function (V::PolyhedralFunction)(x::Vector{Float64})
    return maximum(V.λ * x .+ V.γ)
end

function Base.unique(V::PolyhedralFunction)
    Vu = unique(cat(V.λ, V.γ, dims=2), dims=1)
    return PolyhedralFunction(Vu[:, 1:end-1], Vu[:, end])
end

function add_cut!(V::PolyhedralFunction, λ::Vector{Float64}, γ::Float64)
    if ncuts(V) > 0
        V.λ = cat(V.λ, λ', dims=1)
        push!(V.γ, γ)
    else
        V.λ = λ'
        V.γ = [γ]
    end
    return
end

function remove_cut!(V::PolyhedralFunction, cut_index::Int)
    to_keep = ((1:cut_index-1) ∪ (cut_index+1:ncuts(V)))
    V.λ = V.λ[to_keep, :]
    V.γ = V.γ[to_keep]
    return
end

function remove_cut(V::PolyhedralFunction, cut_index::Int)
    to_keep = (1:cut_index-1) ∪ (cut_index+1:ncuts(V))
    return PolyhedralFunction(V.λ[to_keep, :], V.γ[to_keep])
end

#= 
Given a vector of polyhedral functions V and an array of trajectories, return a new polyhedral function new_V with the cuts of V which are active at some trial point.
=# 

function pruning(V::Vector{PolyhedralFunction}, trajectories::Vector{Matrix{Float64}}; verbose = 0, ε = 1e-6)
    T = length(V)
    nx = dimension(V[1])
    new_V = [PolyhedralFunction(nx) for t in 1:T]
    if verbose > 0
        n_cuts = [ncuts(V[t]) for t in 1:T]
    end
    pruning!(V, trajectories, new_V, T; ε = ε)
    if verbose > 0
        new_n_cuts = [ncuts(new_V[t]) for t in 1:T]
        println("Nb of cuts deleted after pruning: $(sum(n_cuts - new_n_cuts))")
    end
    return new_V
end

function pruning!(V::Vector{PolyhedralFunction}, trajectories::Vector{Array{Float64,2}}, new_V::Vector{PolyhedralFunction}, T::Int; ε=1e-6)
    for t in 1:T
        cut_indexes = zeros(Int, 0)
        for j in 1:length(trajectories)
            if ncuts(V[t]) != 0
                value = V[t](trajectories[j][:, t])
                for i in ncuts(V[t]):-1:1
                    if (i∉ cut_indexes) && (dot(V[t].λ[i, :], trajectories[j][:, t]) + V[t].γ[i] > value - ε)
                        push!(cut_indexes, i)
                    end
                end
            end
        end
        for i in cut_indexes
            add_cut!(new_V[t], V[t].λ[i, :], V[t].γ[i])
        end
    end
end