
#=
    Abstract stochastic model
=#

abstract type AbstractOneStageModel end

abstract type HazardDecisionModel <: AbstractOneStageModel end

function stage_model end

function uncertainties end

function horizon end

function number_states end

function name(hdm::AbstractOneStageModel)
    return "A stochastic model"
end

function Base.show(io::IO, hdm::AbstractOneStageModel)
    nscen = number_nodes(uncertainties(hdm))[end]
    @printf(io, "Problem: %10s\n", name(hdm))
    @printf(io, "    Number of states...: %10i\n", number_states(hdm))
    @printf(io, "    Horizon............: %10i\n", horizon(hdm))
    @printf(io, "    Scenarios..........: %10.2e", nscen)
end

#=
    Multistage Trees
=#


abstract type AbstractNode end

struct Stage{T} <: AbstractNode
    parent::Union{Nothing, Stage{T}}
    model::T
    t::Int
    is_final::Bool
end

abstract type AbstractMultiStageModel end

# Basic multistage problem.
"""
#TODO: Add documentation

"""
struct MultistageProblem{M, N} <: AbstractMultiStageModel
    model::M #TODO rename true_problem ? formal_problem ? problem_description ? problem_type ? 
    stages::Vector{Stage{N}}
end

"""
#TODO: Add documentation
#TODO: change AbstractOneStageModel
"""
function MultistageProblem(model::AbstractOneStageModel)
    T = horizon(model)
    parent = nothing
    stages = Stage{JuMP.Model}[]
    for t in 1:T
        pb = stage_model(model, t)
        is_final = (t == T)
        stage = Stage(parent, pb, t, is_final)
        push!(stages, stage)
        parent = stage
    end
    return MultistageProblem(model, stages)
end

first_stage(pb::MultistageProblem) = pb.stages[1]
final_stage(pb::MultistageProblem) = pb.stages[end]


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

