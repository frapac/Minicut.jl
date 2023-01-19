module Minicut

using Printf
using LinearAlgebra, Statistics
using JuMP
using Dualization

export HazardDecisionModel
export PolyhedralFunction

include("attributes.jl")
include("utils.jl")
include("polyhedral.jl")
include("sddp.jl")
include("extensive_form.jl")
include("dualize.jl")

end # module
