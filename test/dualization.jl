@testset "Dualization" begin
    optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer, "output_flag" => false,
    )
    T, nbins = 3, 2
    wdm = WaterDamModel(T; nbins=nbins)
    Ξ = Minicut.uncertainties(wdm)
    nx = Minicut.number_states(wdm)
    x0 = [8.0]
    t = 1
    # Build primal model
    model = Minicut.stage_model(wdm, t)

    @testset "Dualization: extensive primal model" begin
        moi_model, index_x = Minicut._get_extensive_stage_problem(model, Ξ[t])
        primal_model = JuMP.Model()
        idx_map = MOI.copy_to(primal_model, moi_model)
        @test length(index_x) == nx
        primal_model[Minicut._INITIAL_STATE] = [JuMP.VariableRef(primal_model, idx_map[vi]) for vi in index_x]

        # Solve primal model
        JuMP.set_optimizer(primal_model, optimizer)
        JuMP.fix.(primal_model[Minicut._INITIAL_STATE], x0)
        JuMP.optimize!(primal_model)
        @test JuMP.termination_status(primal_model) == MOI.OPTIMAL
        obj_primal = JuMP.objective_value(primal_model)

        # Test that results match
        obj_ref = 0.0
        JuMP.set_optimizer(model, optimizer)
        for k in 1:length(Ξ[t])
            w = Ξ[t].weights[k]
            ξ = Ξ[t].supports[:, k]
            JuMP.fix.(model[Minicut._PREVIOUS_STATE], x0; force=true)
            JuMP.fix.(model[Minicut._UNCERTAINTIES], ξ)
            JuMP.optimize!(model)
            obj_ref += w * JuMP.objective_value(model)
        end
        @test obj_ref == obj_primal
    end

    @testset "Dualization: extensive dual model" begin
        nw = length(Ξ[t])
        moi_model, index_x = Minicut._get_extensive_stage_problem(model, Ξ[t])
        primal_model = JuMP.Model()
        idx_map = MOI.copy_to(primal_model, moi_model)
        JuMP.set_optimizer(primal_model, optimizer)
        JuMP.optimize!(primal_model)
        obj_primal = JuMP.objective_value(primal_model)

        nvar = JuMP.num_variables(primal_model)
        ncon = JuMP.num_constraints(primal_model; count_variable_in_set_constraints=true)
        dual_model = Minicut.dual_stage_model(wdm, t, -Inf, Inf)
        nvar_dual = JuMP.num_variables(dual_model)
        ncon_dual = JuMP.num_constraints(dual_model; count_variable_in_set_constraints=true)

        # NB: we have added additional variables to account for
        # the co-states μₜ and μₜ₊₁
        @test nvar_dual == ncon + (nx + nw * nx)
        # Co-state [t]
        μ = dual_model[Minicut._PREVIOUS_COSTATE]
        @test isa(μ, Vector{JuMP.VariableRef})
        @test length(μ) == nx
        # Co-state [t+1]
        μf = dual_model[Minicut._CURRENT_COSTATE]
        @test isa(μf, Vector{JuMP.VariableRef})
        @test length(μf) == nx * nw
        # Fix co-state
        JuMP.fix.(μ, [0.0])
        # Solve dual and check we get same objective as in primal
        JuMP.set_optimizer(dual_model, optimizer)
        JuMP.optimize!(dual_model)
        obj_dual = JuMP.objective_value(dual_model)
        @test JuMP.termination_status(dual_model) == MOI.OPTIMAL
        @test obj_dual == obj_primal
    end
end

