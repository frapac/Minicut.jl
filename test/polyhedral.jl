
using Test
using LinearAlgebra
using Minicut

@testset "PolyhedralFunction" begin
    n = 2

    V = Minicut.PolyhedralFunction()

    cut = [1.0, -0.5]
    slope = 1.0
    # Add cut
    Minicut.add_cut!(V, cut, slope)
    x = [2.0, 2.0]
    @test Minicut.ncuts(V) == 1
    @test Minicut.dimension(V) == n
    @test V(x) == dot(x, cut) + slope
    @test size(V.λ) == (1, n)
    @test size(V.γ) == (1, )

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

    # Scenario
    T = 10
    ds = [d for _ in 1:T]
    scenario = Minicut.sample(ds)
    @test isa(scenario, Matrix)
    @test size(scenario) == (n, T)
end

