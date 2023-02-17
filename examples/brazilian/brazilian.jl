#=
    Brazilian Hydro unit-commitment
=#

using DelimitedFiles
using LinearAlgebra
using Random, Statistics
using JuMP
using HiGHS
using Minicut

DATA_FOLDER = joinpath(@__DIR__, "data")

function import_raw_data(name)
    return readdlm(joinpath(DATA_FOLDER, name), ',', header=true)[1][:, 2:end] .|> Float64
end

struct BrazilianHydroData
    hydro::Matrix{Float64}
    demand::Matrix{Float64}
    deficit::Matrix{Float64}
    exchange_ub::Matrix{Float64}
    exchange_costs::Matrix{Float64}
    thermals::Vector{Matrix{Float64}}
end

function BrazilianHydroData()
    return BrazilianHydroData(
        import_raw_data("hydro.csv"),
        import_raw_data("demand.csv"),
        import_raw_data("deficit.csv"),
        import_raw_data("exchange.csv"),
        import_raw_data("exchange_cost.csv"),
        [import_raw_data("thermal_$i.csv") for i in 0:3],
    )
end

data = BrazilianHydroData()

struct BrazilianHydroModel <: HazardDecisionModel
    T::Int
    data::BrazilianHydroData
    xmax::Vector{Float64}
    x0::Vector{Float64}
    uturb_max::Vector{Float64}
    inflows::Vector{Minicut.DiscreteRandomVariable{Float64}}
end

function import_scenarios(nstages, nscenarios, inflow_initial)
    scenarios = zeros(nstages, nscenarios, 4)
    for k in 1:4
        _scen = readdlm(joinpath(DATA_FOLDER, "scenario_inflows_$(k-1).txt"))
        scenarios[2:nstages, :, k] .= _scen[1:nstages-1, 1:nscenarios]
    end

    uncertainties = Minicut.DiscreteRandomVariable{Float64}[]
    w0, ξ0 = [1.0], reshape(inflow_initial, 4, 1)
    push!(uncertainties, Minicut.DiscreteRandomVariable{Float64}(w0, ξ0))
    for t in 2:nstages
        weights = ones(nscenarios) ./ nscenarios
        supports = scenarios[t, :, :]'
        push!(uncertainties, Minicut.DiscreteRandomVariable{Float64}(weights, supports))
    end
    return uncertainties
end

function BrazilianHydroModel(; T=12, nscen=10)
    data = BrazilianHydroData()
    inflow_initial = data.hydro[5:8, 2]
    return BrazilianHydroModel(
        T,
        data,
        data.hydro[1:4, 1],
        data.hydro[1:4, 2],
        data.hydro[9:12, 1],
        import_scenarios(T, nscen, inflow_initial),
    )
end

Minicut.uncertainties(bhm::BrazilianHydroModel) = bhm.inflows
Minicut.horizon(bhm::BrazilianHydroModel) = bhm.T
Minicut.number_states(bhm::BrazilianHydroModel) = 4
Minicut.name(bhm::BrazilianHydroModel) = "Brazilian hydro-thermal generation problem"

function Minicut.stage_model(bm::BrazilianHydroModel, t::Int)
    discount = 0.9906
    β = discount^(t-1)
    cost_spill = 1e-3
    demand = data.demand[(t-1) % 12 + 1, :]
    exch_costs = bm.data.exchange_costs
    m = Model()

    @variable(m, 0.0 <= dams[i=1:4] <= bm.xmax[i])
    @variable(m, 0.0 <= damsf[i=1:4] <= bm.xmax[i])
    @variable(m, 0.0 <= uturb[i=1:4] <= bm.uturb_max[i])
    @variable(m, 0.0 <= uspill[i=1:4] <= 100000)

    @variable(m, 0.0 <= deficit[i=1:4, j=1:4] <= demand[i] * bm.data.deficit[j, 2])
    @variable(m, 0.0 <= exch[i=1:5, j=1:5] <= bm.data.exchange_ub[i, j])

    @variable(m, inflows[i=1:4])

    utherm = Dict{Int,Array{VariableRef,1}}()
    for k in 1:4
        therm = bm.data.thermals[k]
        nth = size(therm, 1)
        utherm[k] = @variable(m, [i=1:nth], lower_bound=therm[i, 1], upper_bound=therm[i, 2])
    end

    # Dynamics
    for k in 1:4
        @constraint(m, damsf[k] == dams[k] - uturb[k] - uspill[k] + inflows[k])
    end

    # Exchange
    for k in 1:4
        @constraint(m,
            sum(utherm[k])
            + sum(deficit[k, j] for j in 1:4)
            + uturb[k]
            - sum(exch[k, j] for j in 1:4)
            + sum((1.0 - exch_costs[j, k]) * exch[j, k] for j in 1:4) == demand[k]
        )
    end
    ## At residual node
    @constraint(m, sum(exch[j, 5] for j in 1:5) == sum(exch[5, j] for j in 1:5))

    # Objective
    @objective(m, Min,
        β * (
            cost_spill * sum(uspill) +
            sum(dot(bm.data.thermals[k][:, 3], utherm[k]) for k in 1:4)
            + sum(bm.data.deficit[j, 1] * deficit[i, j] for i in 1:4, j in 1:4)
        )
    )

    @expression(m, x₋, dams)
    @expression(m, x, damsf)
    @expression(m, u, [uturb; uspill; utherm[1] ; utherm[2] ; utherm[3] ; utherm[4]; deficit[:]; exch[:]])
    @expression(m, ξ, inflows)

    return m
end

function brazilian(; T=12, max_iter=500, nscenarios=10, nsimus=1000)
    Random.seed!(2713)

    bhm = BrazilianHydroModel(; T=T, nscen=nscenarios)
    nx = Minicut.number_states(bhm)
    x0 = bhm.x0

    # Initialize value functions.
    lower_bound = -1e6
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]

    # Solve with SDDP
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    solver = Minicut.SDDP(optimizer, [MOI.OPTIMAL, MOI.OTHER_ERROR])
    models = Minicut.solve!(solver, bhm, V, x0; n_iter=max_iter, verbose=10)

    # Simulation
    scenarios = Minicut.sample(Minicut.uncertainties(bhm), nsimus)
    costs = Minicut.simulate!(solver, bhm, models, x0, scenarios)
    ub = mean(costs) + 1.96 * std(costs) / sqrt(nsimus)
    lb = V[1](x0)

    println("Final statistical gap: ", abs(ub - lb) / abs(ub))
    return
end

