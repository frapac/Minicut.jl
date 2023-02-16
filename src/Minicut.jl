module Minicut

using Printf
using Random
using LinearAlgebra, Statistics
using JuMP
using Dualization

export HazardDecisionModel
export PolyhedralFunction

include("attributes.jl")
include("utils.jl")
include("polyhedral.jl")
include("extensive_form.jl")
include("dualize.jl")
include("SDDP/generic_sddp.jl")

end # module
