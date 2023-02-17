
struct WaterDamModel <: HazardDecisionModel
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
    @variable(m, 0 <= turb <= wdm.umax)
    @variable(m, 0 <= spill <= 10000.0)

    @constraint(m, l2 == l1 - turb + r2 - spill)

    @objective(m, Min, -wdm.csell[t] * turb)

    @expression(m, x₋, [l1])
    @expression(m, u, [turb, spill])
    @expression(m, x, [l2])
    @expression(m, ξ, [r2])

    return m
end

