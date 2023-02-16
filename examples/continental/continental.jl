#=
    CONTINENTAL example.
=#

using DelimitedFiles
using LinearAlgebra
using Random, Statistics
using JuMP
using HiGHS
using Minicut

DATA_FOLDER = joinpath(@__DIR__, "data", "scen")

struct ContinentalData
    order::Vector{Symbol}
    connection::Matrix{Float64}
    # Stocks
    xmin::Vector{Float64}
    xmax::Vector{Float64}
    x0::Vector{Float64}
    # Controls
    uturb_max::Vector{Float64}
    utherm_max::Vector{Float64}
    # Costs
    ctherm_pl::Matrix{Float64}
    cexch::Float64
    cpenal::Float64
    cfinal::Float64
end

function ContinentalData()
    monthly_ratio = 30 * 24
    A = monthly_ratio*Float64[
        0 0 2 1 0 0 0 0;
        0 0 1 0 0 1 0 0;
        2 1 0 3 1 0 2 1;
        1 0 3 0 0 0 1 0;
        0 0 1 0 0 0 1 0;
        0 1 0 0 0 0 0 0;
        0 0 2 1 1 0 0 0;
        0 0 1 0 0 0 0 0
    ]
    return ContinentalData(
        [:BEL, :ESP, :FRA, :GER, :ITA, :PT, :SUI, :UK],
        A,
        1000.0*Float64[0, 0, 0,   0, 0, 0,   0,   0],
        1000.0*Float64[0, 8, 10,  4, 4, 2,   10,  2],
        1000.0*Float64[0, 6, 7.5, 3, 3, 1.5, 7.5, 1.5],
        monthly_ratio .* Float64[0, 9, 12,  3, 10.5, 1.5, 12,  1.5],
        monthly_ratio .* Float64[40, 40, 100, 100, 40, 20, 20, 60],
        Float64[
            10 40 70 90;
            10 40 70 90;
            5  15 30 45;
            10 25 35 50;
            10 40 70 90;
            20 100 100 100;
            20 100 100 100;
            20 40 60 80
        ],
        1.0, 3000.0, 3000.0,
    )
end

function import_scenarios(names, nstages, nscen)
    sep = ','
    scaling = 24.0
    nzones = length(names)
    nξ = 2 * nzones
    scenarios = zeros(nstages, nscen, nξ)
    for (k, name) in enumerate(names)
        # inflow (in GWh)
        _inflow = scaling .* readdlm(joinpath(DATA_FOLDER, "$name/inflow.txt"), sep)[1:nscen, 1:nstages]
        # demand (in GWh)
        _demand = scaling .* readdlm(joinpath(DATA_FOLDER, "$name/demands.txt"), sep)[1:nscen, 1:nstages]

        scenarios[:, :, k] .= _inflow'
        scenarios[:, :, k+nzones] .= _demand'
    end

    uncertainties = Minicut.DiscreteRandomVariable{Float64}[]
    for t in 1:nstages
        weights = ones(nscen) ./ nscen
        supports = scenarios[t, :, :]'
        push!(uncertainties, Minicut.DiscreteRandomVariable{Float64}(weights, supports))
    end
    return uncertainties
end

struct ContinentalModel <: HazardDecisionModel
    T::Int
    nzones::Int
    nedges::Int
    incidence::Matrix{Float64}
    # States
    x0::Vector{Float64}
    xmax::Vector{Float64}
    # Controls
    uturb_max::Vector{Float64}
    utherm_max::Vector{Float64}
    qmax::Vector{Float64}
    # Costs
    ctherm::Matrix{Float64}
    cpenal::Float64
    cexch::Float64
    cfinal::Float64
    uncertainties::Vector{Minicut.DiscreteRandomVariable{Float64}}
end

function ContinentalModel(
    names, T; nscenarios=10,
)
    nzones = length(names)
    # Raw data
    data = ContinentalData()
    position = [findfirst(data.order .== name) for name in names]

    # Build incidence matrix
    C = data.connection[position, position]
    nedges = div(sum(C .> 0.0), 2)
    A = zeros(Float64, nzones, nedges)
    ic = 0
    qmax = Float64[]
    for ix in 1:(nzones-1), iy in (ix+1):nzones
        if C[ix, iy] > 0
            ic += 1
            A[ix, ic] =  1
            A[iy, ic] = -1
            push!(qmax, C[ix, iy])
        end
    end

    # Costs
    ctherm = data.ctherm_pl[position, 1] .+ 15.0 .* rand(nzones, T)

    uncertainties = import_scenarios(names, T, nscenarios)

    return ContinentalModel(
        T, nzones, nedges,
        A,
        data.x0[position],
        data.xmax[position],
        data.uturb_max[position],
        data.utherm_max[position],
        qmax,
        ctherm,
        data.cpenal,
        data.cexch,
        data.cfinal,
        uncertainties,
    )
end

Minicut.uncertainties(cm::ContinentalModel) = cm.uncertainties
Minicut.horizon(cm::ContinentalModel) = cm.T
Minicut.number_states(cm::ContinentalModel) = cm.nzones

function Minicut.stage_model(cm::ContinentalModel, t::Int)
    nz = cm.nzones
    na = cm.nedges

    m = Model()

    @variable(m, 0.0 <= x[i=1:nz] <= cm.xmax[i])
    @variable(m, 0.0 <= xf[i=1:nz] <= cm.xmax[i])
    @variable(m, 0.0 <= uturb[i=1:nz] <= cm.uturb_max[i])
    @variable(m, 0.0 <= uspill[1:nz] <= 10000.0)
    @variable(m, 0.0 <= utherm[i=1:nz] <= cm.utherm_max[i])
    @variable(m, 0.0 <= urecourse[1:nz] <= 10000.0)
    @variable(m, -cm.qmax[i] <= flows[i=1:na] <= cm.qmax[i])

    @variable(m, inflows[1:nz])
    @variable(m, demands[1:nz])

    # Dynamics
    for n in 1:nz
        @constraint(m, xf[n] == x[n] - uturb[n] - uspill[n] + inflows[n])
    end

    # Balance equations
    @expression(m, exch, cm.incidence * flows)
    for n in 1:nz
        @constraint(m, uturb[n] + utherm[n] + urecourse[n] + exch[n] == demands[n])
    end

    # Objective
    if t < Minicut.horizon(cm)
        @objective(m, Min, dot(cm.ctherm[:, t], utherm) + cm.cpenal * sum(urecourse) + cm.cexch * sum(flows))
    elseif t == Minicut.horizon(cm)
        @variable(m, K[1:nz] >= 0.0)
        @constraint(m, K .>= cm.x0 .- xf)
        @objective(
            m,
            Min,
            dot(cm.ctherm[:, t], utherm) + cm.cpenal * sum(urecourse) + cm.cexch * sum(flows) + cm.cfinal * sum(K)
        )
    end

    @expression(m, xₜ, x)
    @expression(m, uₜ₊₁, [uturb; uspill; utherm; urecourse; flows])
    @expression(m, xₜ₊₁, xf)
    @expression(m, ξₜ₊₁, [inflows ; demands])

    return m
end

function continental(; names=[:FRA, :GER, :ESP, :PT, :ITA, :SUI, :UK, :BEL], T=12, max_iter=500, nscenarios=10, nsimus=1000)
    Random.seed!(2713)

    cm = ContinentalModel(names, T; nscenarios=nscenarios)
    nx = Minicut.number_states(cm)
    x0 = cm.x0

    # Initialize value functions.
    lower_bound = -1e6
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]

    # Solve with SDDP
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    solver = Minicut.SDDP(optimizer)
    models = Minicut.solve!(solver, cm, V, x0; n_iter=max_iter, verbose=10)

    # Simulation
    scenarios = Minicut.sample(Minicut.uncertainties(cm), nsimus)
    costs = Minicut.simulate!(solver, cm, models, x0, scenarios)
    ub = mean(costs) + 1.96 * std(costs) / sqrt(nsimus)
    lb = V[1](x0)

    println("Final statistical gap: ", abs(ub - lb) / abs(ub))
    return
end

