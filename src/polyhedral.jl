mutable struct PolyhedralFunction
    λ::Array{Float64, 2}
    γ::Array{Float64, 1}
end

ncuts(V::PolyhedralFunction) = size(V.λ, 1)
dimension(V::PolyhedralFunction) = size(V.λ, 2)
eachcut(V::PolyhedralFunction) = zip(eachrow(V.λ), V.γ)

function PolyhedralFunction()
    return PolyhedralFunction(Array{Float64,2}(undef, 0, 0), Vector{Float64}())
end

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

# TODO
function fenchel_transform_as_sup(m::JuMP.Model, D::PolyhedralFunction, x::Array{Float64,1}, lip::Real)
    nx = size(D.λ, 2)
    @variable(m, -lip <= λ[1:nx] <= lip)
    @variable(m, θ)
    for (xk, βk) in eachcut(D)
        @constraint(m, θ >= xk' * λ + βk)
    end
    @objective(m, Max, x' * λ - θ)
end

# TODO
function fenchel_transform_as_inf(m::JuMP.Model, D::PolyhedralFunction, x::Array{Float64,1}, lip::Real)
    nc, nx = size(D.λ)
    @variable(m, σ[1:nc] >= 0)
    @constraint(m, sum(σ) == 1)
    @variable(m, y[1:nx])
    @variable(m, lift[1:nx] >= 0)
    @constraint(m, lift .>= x .- y)
    @constraint(m, lift .>= y .- x)
    @constraint(m, sum(σk .* D.λ[k, :] for (k, σk) in enumerate(σ)) .== y)
    @objective(m, Min, lip * sum(lift) - sum(σ .* D.γ))
end

