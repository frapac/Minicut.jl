include("../examples/brazilian/brazilian.jl")


@testset "Tree" begin
    T = 12
    n_scenarios = 50
    n_sample = 10
    Random.seed!(78964)

    bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
    Ξ = Minicut.uncertainties(bhm)
    ξ = Minicut.sample(Ξ)
    scenarios = Minicut.sample(Ξ, n_sample)
    fake_scenario = zeros(Float64, 4, T)
    fake_scenario[:,1:6] = scenarios[3][:, 1:6]
    fake_scenario[:,7:T] = scenarios[1][:, 7:T]
    push!(scenarios, fake_scenario)
    @test Minicut.scenario_path(bhm, ξ, Ξ) == [1, 37, 45, 7, 24, 8, 17, 49, 29, 24, 6, 27]
    @test Minicut.scenario_path(bhm, scenarios[3], Ξ)[1:6] == Minicut.scenario_path(bhm, fake_scenario, Ξ)[1:6]

    @test Minicut.weight(bhm, ξ, Ξ) ≈ 2.048*1e-19

    # The scenarios 3 and 11 (= fake_scenario) have common history up to t=6, scenario 7 also go through same node (6,4) but has a different history
    @test Minicut.histories_tj(bhm, scenarios, Ξ, 6, 4) == ([[3, 11], [7]], [[1, 41, 35, 18, 32, 4], [1, 9, 50, 16, 39, 4]])


    #    @test Minicut.cost_tj!(bhm, scenarios, )
    
end