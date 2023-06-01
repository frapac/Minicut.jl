
#=
    Abstract stochastic model
=#

abstract type AbstractBellmanModel end

abstract type HazardDecisionModel <: AbstractBellmanModel end

function stage_model end

function uncertainties end

function horizon end

function number_states end

function name(hdm::AbstractBellmanModel)
    return "A stochastic model"
end

function Base.show(io::IO, hdm::AbstractBellmanModel)
    nscen = number_nodes(uncertainties(hdm))[end]
    @printf(io, "Problem: %10s\n", name(hdm))
    @printf(io, "    Number of states...: %10i\n", number_states(hdm))
    @printf(io, "    Horizon............: %10i\n", horizon(hdm))
    @printf(io, "    Scenarios..........: %10.2e", nscen)
end

#=
    Abstract stochastic optimizer
=#

abstract type AbstractStochasticOptimizer end

function introduce end

#=
    Keys
=#
const _INITIAL_STATE         = :(x0)
const _PREVIOUS_STATE        = :(x₋)
const _CURRENT_STATE         = :(x)
const _CURRENT_CONTROL       = :(u)
const _UNCERTAINTIES         = :(ξ)
const _PREVIOUS_COSTATE      = :(μ₋)
const _CURRENT_COSTATE       = :(μ)
const _VALUE_FUNCTION        = :(θ)



