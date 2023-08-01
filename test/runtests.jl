
using Test
using LinearAlgebra
using JuMP
using Minicut

# Temporary
using Gurobi
const GRB_ENV = Gurobi.Env(output_flag = 0)
include("../examples/brazilian/brazilian.jl")

include("models.jl")
include("tree.jl")
include("polyhedral.jl")
include("weird_regularized.jl")
#include("extensive_formulation.jl")
# include("sddp.jl")
# include("dualization.jl")
# include("dual_sddp.jl")
# include("interface.jl")

