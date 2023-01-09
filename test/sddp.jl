using JuMP
using HiGHS
using Statistics

struct WaterDamModel <: HereAndNowModel
    T::Int
    capacity::Float64
    umax::Float64
    csell::Vector{Float64}
    inflows::Vector{Minicut.DiscreteRandomVariable{Float64}}
end

function WaterDamModel(
    T;
    nbins=10,
    capacity=10.0,
    umax=5.0,
    maxflow =3.0,
    csell=120.0*(1.0 .+ 0.5 .* (rand(T) .- 0.5)),
)
    # Build uncertainty model
    weights = 1.0 ./ nbins .* ones(nbins)
    inflows = [Minicut.DiscreteRandomVariable(weights, maxflow .* rand(1, nbins)) for t in 1:T]
    return WaterDamModel(T, capacity, umax, csell, inflows)
end

Minicut.uncertainties(wdm::WaterDamModel) = wdm.inflows
Minicut.horizon(wdm::WaterDamModel) = wdm.T
Minicut.number_states(wdm::WaterDamModel) = 1

function Minicut.stage_model(wdm::WaterDamModel, t::Int)
    m = Model()

    @variable(m, 0 <= l1 <= wdm.capacity)
    @variable(m, 0 <= l2 <= wdm.capacity)
    @variable(m, r2)
    @variable(m, 0 <= u <= wdm.umax)
    @variable(m, s >= 0)

    @constraint(m, l2 == l1 - u + r2 - s)

    @objective(m, Min, -wdm.csell[t] * u)

    @expression(m, xₜ, [l1])
    @expression(m, uₜ₊₁, [u, s])
    @expression(m, xₜ₊₁, [l2])
    @expression(m, ξₜ₊₁, [r2])

    return m
end

@testset "SDDP: WaterDamModel" begin
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    lower_bound = -1e4
    x0 = [8.0]
    T = 3
    nx = 1
    wdm = WaterDamModel(T)
    Ξ = Minicut.uncertainties(wdm)

    @test Minicut.horizon(wdm) == T

    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
    # Final value function
    push!(V, Minicut.PolyhedralFunction(zeros(1, nx), [0.0]))

    #=
        First, we test the mechanics of the forward and backward passes.
    =#
    @testset "Forward/backward pass" begin
        # JuMP Model
        models = JuMP.Model[Minicut.stage_model(wdm, t) for t in 1:T]
        for (t, Vₜ₊₁) in enumerate(V[2:end])
            Minicut.initialize!(models[t], Vₜ₊₁)
            JuMP.set_optimizer(models[t], optimizer)
        end

        scenario = Minicut.sample(Ξ)

        primal_scenarios = Minicut.forward_pass(wdm, models, scenario, x0)

        @test isa(primal_scenarios, Array{Float64, 2})
        @test size(primal_scenarios) == (nx, T+1)
        # Test JuMP model is well defined
        @test JuMP.value.(models[1][:xₜ]) == x0
        @test JuMP.value.(models[1][:ξₜ₊₁]) == scenario[:, 1]

        dual_scenarios = Minicut.backward_pass!(wdm, models, primal_scenarios, V)
        @test size(dual_scenarios) == size(primal_scenarios)
        # Test we have the right amount of cuts.
        @test Minicut.ncuts(V[1]) == 2
    end

    #=
        Test SDDP itself.
    =#
    @testset "SDDP algorithm" begin
        solver = Minicut.SDDP(optimizer)
        models = Minicut.solve!(solver, wdm, V, x0; n_iter=100, verbose=0)

        n_scenarios = 1000
        scenarios = Minicut.sample(Ξ, n_scenarios)
        costs = Minicut.simulate!(wdm, models, x0, scenarios)

        lb = V[1](x0)
        ub = mean(costs) + 1.96 * std(costs) / sqrt(n_scenarios)
        # Test convergence of SDDP
        @test abs(ub - lb) / abs(ub) <= 0.01
    end

    # Test lower bound matches exactly upper bound in deterministic case
    @testset "Deterministic problem" begin
        T = 10
        Vdet = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
        push!(Vdet, Minicut.PolyhedralFunction(zeros(1, nx), [0.0]))

        wdm_determistic = WaterDamModel(T; nbins=1)
        solver = Minicut.SDDP(optimizer)
        models = Minicut.solve!(solver, wdm_determistic, Vdet, x0; n_iter=100, verbose=0)
        #
        n_scenarios = 1
        scenarios = Minicut.sample(Minicut.uncertainties(wdm_determistic), n_scenarios)
        costs = Minicut.simulate!(wdm_determistic, models, x0, scenarios)
        @test mean(costs) ≈ Vdet[1](x0)
    end
end

