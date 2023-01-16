#=
    A simple damsvalley problem.

    Optimize the dispatch for 5 dams.

=#

using Random, Statistics
using JuMP
using HiGHS
using Minicut

struct DamsValleyModel <: HazardDecisionModel
    T::Int
    capacity::Float64
    umax::Float64
    smax::Float64
    csell::Vector{Float64}
    inflows::Vector{Minicut.DiscreteRandomVariable{Float64}}
end

function DamsValleyModel(
    T::Int; vmax=80.0, umax=40.0, smax=200.0, maxflow=9.0, nbins=10,
)
    weights = ones(nbins) ./ nbins
    inflows = [Minicut.DiscreteRandomVariable(weights, maxflow .* rand(5, nbins)) for t in 1:T]
    csell = 178.2*(1.0 .+ 0.5*(rand(T) .- 0.5))
    return DamsValleyModel(T, vmax, umax, smax, csell, inflows)
end

Minicut.uncertainties(dvm::DamsValleyModel) = dvm.inflows
Minicut.horizon(dvm::DamsValleyModel) = dvm.T
Minicut.number_states(dvm::DamsValleyModel) = 5

function Minicut.stage_model(dvm::DamsValleyModel, t::Int)
    m = Model()

    @variable(m, 0 <= x[1:5] <= dvm.capacity)
    @variable(m, 0 <= xf[1:5] <= dvm.capacity)
    @variable(m, 0 <= u[1:5] <= dvm.umax)
    @variable(m, 0 <= s[1:5] <= dvm.smax)
    @variable(m, inflow[1:5])

    # Dynamics
    # dam1 -> dam2 -> dam3 -> dam4 -> dam5
    @constraints(m, begin
        xf[1] == x[1] - u[1] - s[1] + inflow[1]
        xf[2] == x[2] + u[1] + s[1] - u[2] - s[2] + inflow[2]
        xf[3] == x[3] + u[2] + s[2] - u[3] - s[3] + inflow[3]
        xf[4] == x[4] + u[3] + s[3] - u[4] - s[4] + inflow[4]
        xf[5] == x[5] + u[4] + s[4] - u[5] - s[5] + inflow[5]
    end)

    if t < Minicut.horizon(dvm)
        @objective(m, Min, -dvm.csell[t] * sum(u))
    elseif t == Minicut.horizon(dvm)
        @variable(m, K[1:5] >= 0.0)
        @constraint(m, K .>= 40.0 .- xf)
        @objective(m, Min, -dvm.csell[t] * sum(u) + 500.0 * sum(K))
    end

    @expression(m, xₜ, x)
    @expression(m, uₜ₊₁, [u; s])
    @expression(m, xₜ₊₁, xf)
    @expression(m, ξₜ₊₁, inflow)

    return m
end

function damsvalley(; max_iter=500, nbins=10, nsimus=1000)
    Random.seed!(2713)
    T = 12

    dvm = DamsValleyModel(T; nbins=nbins)
    nx = Minicut.number_states(dvm)
    x0 = fill(40.0, nx)

    # Initialize value functions.
    lower_bound = -1e6
    V = [Minicut.PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
    push!(V, Minicut.PolyhedralFunction(zeros(1, nx), [0.0]))

    # Solve with SDDP
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    solver = Minicut.SDDP(optimizer)
    models = Minicut.solve!(solver, dvm, V, x0; n_iter=max_iter, verbose=10)

    # Simulation
    scenarios = Minicut.sample(Minicut.uncertainties(dvm), nsimus)
    costs = Minicut.simulate!(dvm, models, x0, scenarios)
    ub = mean(costs) + 1.96 * std(costs) / sqrt(nsimus)
    lb = V[1](x0)

    println("Final gap: ", abs(ub - lb) / abs(ub))

    return models
end

