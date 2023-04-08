mutable struct PolyhedralFunction
    λ::Array{Float64, 2}
    γ::Array{Float64, 1}
end

ncuts(V::PolyhedralFunction) = size(V.λ, 1)
dimension(V::PolyhedralFunction) = size(V.λ, 2)
eachcut(V::PolyhedralFunction) = zip(eachrow(V.λ), V.γ)

PolyhedralFunction(nx::Int) = PolyhedralFunction(zeros(0, nx), Float64[])
PolyhedralFunction(nx::Int, lb::Float64) = PolyhedralFunction(zeros(1, nx), [lb])

function lipschitz_constant(V::PolyhedralFunction, pnorm::Real = 1)
    return maximum([norm(λ, pnorm) for λ in eachrow(V.λ)])
end

function (V::PolyhedralFunction)(x::Vector{Float64})
    return maximum(V.λ * x .+ V.γ)
end

function Base.unique(V::PolyhedralFunction)
    Vu = unique(cat(V.λ, V.γ, dims = 2), dims = 1)
    return PolyhedralFunction(Vu[:, 1:end-1], Vu[:, end])
end

function add_cut!(V::PolyhedralFunction, λ::Vector{Float64}, γ::Float64)
    if ncuts(V) > 0
        V.λ = cat(V.λ, λ', dims = 1)
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
Given two vectors of polyhedral functions V and V_ref, given a trajectory,
add to V the active cuts of V_ref at the trajectory
=#

function prunning!(V::Vector{PolyhedralFunction}, V_ref::Vector{PolyhedralFunction}, trajectory::Array{Float64,2}; ε = 1e-6)
    T = length(V)
    for t in 1:T
        value = V_ref[t](trajectory[:, t])
        for i in 1:ncuts(V_ref[t])
            if dot(V_ref[t].λ[i,:], trajectory[:, t]) + V_ref[t].γ[i] < value - ε
                add_cut!(V[t], V_ref[t].λ[i,:], V_ref[t].γ[i])
            end
        end
    end
end

