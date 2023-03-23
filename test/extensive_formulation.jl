
using JuMP
using HiGHS
using Statistics
using Minicut

@testset "Extensive formulation: WaterDamModel" begin
    T = 3
    wdm = WaterDamModel(T; nbins = 2)
    nx = Minicut.number_states(wdm)
    Ξ = Minicut.uncertainties(wdm)
    x0 = [8.0]

    ext = Minicut.build_scenario_tree(wdm)

    @test ext.moi_model isa MOI.AbstractOptimizer
    # Test we have the correct number of nodes
    nnodes = Minicut.number_nodes(Ξ)
    @test sum(nnodes) == length(ext.scenario_tree)

    m = JuMP.Model()
    MOI.copy_to(m, ext.moi_model)

    optimizer = JuMP.optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
    solver = Minicut.ExtensiveFormulationSolver(optimizer)

    model = Minicut.solve!(solver, wdm, x0)
    @test isa(model, JuMP.Model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
end
