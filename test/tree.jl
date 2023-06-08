include("../examples/brazilian/brazilian.jl")

@testset "Tree" begin
    T = 12
    n_scenarios = 50
    Random.seed!(78964)

    bhm = BrazilianHydroModel(; T=T, nscen=n_scenarios)
    Ξ = Minicut.uncertainties(bhm)

    ξ = Minicut.sample(Ξ)

    @test Minicut.scenario_path(bhm, ξ, Ξ) == [1, 37, 45, 7, 24, 8, 17, 49, 29, 24, 6, 27]

    println(Minicut.weight(bhm, ξ, Ξ))
end