
using Test
using LinearAlgebra
using Minicut

@testset "PolyhedralFunction" begin
    n = 2

    V = Minicut.PolyhedralFunction(n)

    cut = [1.0, -0.5]
    slope = 1.0
    # Add cut
    Minicut.add_cut!(V, cut, slope)
    x = [2.0, 2.0]
    @test Minicut.ncuts(V) == 1
    @test Minicut.dimension(V) == n
    @test V(x) == dot(x, cut) + slope
    @test size(V.λ) == (1, n)
    @test size(V.γ) == (1,)

    # Cut removal
    Vu = unique(V)
    @test isa(Vu, Minicut.PolyhedralFunction)
    @test Minicut.ncuts(Vu) == 1

    Minicut.remove_cut!(V, 2)
    @test Minicut.ncuts(V) == 1
    @test Minicut.lipschitz_constant(V, Inf) == 1.0
end

@testset "DiscreteRandomVariable" begin
    n, m = 10, 3
    weights = [0.1, 0.7, 0.2]
    supports = randn(n, m)
    d = Minicut.DiscreteRandomVariable(weights, supports)

    @test Minicut.dimension(d) == n
    @test Minicut.length(d) == m
    @test isa(rand(d), Vector)
    k = 2
    v = d.supports[:, k]
    @test Minicut.find_outcome(d, v) == k

    # Scenario
    T = 10
    ds = [d for _ in 1:T]
    scenario = Minicut.sample(ds)
    @test isa(scenario, Matrix)
    @test size(scenario) == (n, T)
end

@testset "Pruning" begin
    T = 3
    nx = 3
    lower_bound = 0.0
    # V[1] = V[2] = V[3] = "the absolute value", tested at -1, 0 and 1. We want to keep -x at t=0, both at t=1 and +x at t=2. We should throw away two cuts.
    trajectory = hcat(-ones(nx, 1), zeros(nx, 1), ones(nx, 1))
    trajectories = [trajectory, trajectory]
    V = [Minicut.PolyhedralFunction(vcat(ones(1, nx), -ones(1, nx)), [lower_bound, lower_bound]) for t in 1:T]
    V = Minicut.pruning(V::Vector{PolyhedralFunction}, trajectories; ε=1e-6, verbose = 0)
    @test V[1].λ == -ones(1, nx)
    @test V[1].γ == [0.0]
    @test V[2].λ == [ -1.0 -1.0 -1.0; 1.0 1.0 1.0]
    @test V[2].γ == [0.0, 0.0]
    @test V[3].λ == ones(1, nx)
    @test V[3].γ == [0.0]
end
