
#=
    Regularized primal SDDP
=#

struct RegularizedStage{M} <: AbstractRegularizedNode
    parent::Union{Nothing, RegularizedStage{M}}
    model::M
    regularized_model::M
    t::Int
    is_final::Bool
end

struct RegularizedPrimalSDDP <: AbstractRegularizedSDDP
    qp_optimizer::Any
    primal_sddp::SDDP
    dual_sddp::DualSDDP
    tau::Float64
    mixing::Float64
end

introduce(::RegularizedPrimalSDDP) = "Regularized Primal SDDP"

function lowerbound(
    sddp::SDDP,
    stage::AbstractNode,
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64},
)
    solve_stage_problem!(sddp, stage, xₜ, ξₜ₊₁)
    return JuMP.objective_value(stage.model)
end

function upperbound(
    dual_sddp::DualSDDP,
    model::JuMP.Model,
    xₜ::Vector{Float64},
    ξₜ₊₁::Vector{Float64},
    D::PolyhedralFunction,
)
    fix.(model[_PREVIOUS_STATE], xₜ, force=true)
    fix.(model[_UNCERTAINTIES], ξₜ₊₁, force=true)
    JuMP.set_optimizer(model, dual_sddp.optimizer)
    # future states
    xf = model[_CURRENT_STATE]
    # number of cuts
    n_cuts = ncuts(D)
    # Lipschitz constant
    lipschitz = dual_sddp.lipschitz_ub

    # define simplex Λ
    @variable(model, eta[1:n_cuts] >= 0.0)
    @variable(model, xabs[1:size(xf)[1]])
    @variable(model, x_alt[1:size(xf)[1]])

    @constraint(model, sum(eta) == 1.0)
    # we build the inner approximation all in once
    @constraint(model, xabs .>= xf - x_alt) #norm1
    @constraint(model, xabs .>= x_alt - xf)
    @constraint(model, x_alt .== sum(eta[i] * D.λ[i, :] for i in 1:ncuts(D)))

    cost_fct = objective_function(model)
    @objective(model, Min, cost_fct - sum(eta[i] * D.γ[i] for i in 1:n_cuts) + lipschitz * sum(xabs))
    optimize!(model)
    return objective_value(model)
end

function level_parameter(reg::RegularizedPrimalSDDP, lb::Float64, ub::Float64)
    relative_gap = (ub-lb)/abs(lb)
    if lb - 1e-3 > ub 
        println("abs(ub-lb) = $(abs(ub-lb))")
        @assert lb - 1e-3 <= ub 
    end
    if relative_gap < 0.12
        #mixing = relative_gap
        mixing = .9
        ℓ = mixing * lb + (1.0 - mixing) * ub
    else
        ℓ = 0.98*lb 
        # ℓ = -1e9
    end
    # ℓ = -1e9

    return ℓ
end 

function forward_pass!(
    reg_sddp::RegularizedPrimalSDDP,
    tree::MultistageProblem,
    V::Vector{PolyhedralFunction},
    D::Vector{PolyhedralFunction},
    scenario::InSampleScenario{Float64},
    initial_state::Vector{Float64},
    trajectory::Array{Float64,2},
)
    Ξ = uncertainties(tree.model)
    xₜ = copy(initial_state)
    trajectory[:, 1] .= xₜ
    level_cnt = 0
    solve_cnt = 0
    for stage in tree.stages
        wₜ = scenario.values[:, stage.t]
        # Lower-bound.
        lb = lowerbound(reg_sddp.primal_sddp, stage, xₜ, wₜ)

        # Upper-bound.
        if !stage.is_final
            ubmodel = stage_model(tree.model, stage.t)
            ub = upperbound(reg_sddp.dual_sddp, ubmodel, xₜ, wₜ, D[stage.t+1])
        else
            ub = lb
        end

        # Regularization level ; Adaptative combination between lb and ub depending on the relative gap
        ℓ = level_parameter(reg_sddp, lb, ub)
        xₜ, is_level = next!(reg_sddp, stage, xₜ, Ξ[stage.t], wₜ, ℓ)
        solve_cnt += 1
        level_cnt += is_level

        trajectory[:, stage.t+1] .= xₜ
    end
    return trajectory, solve_cnt, level_cnt
end

function forward_pass(
    sddp::RegularizedPrimalSDDP,
    tree::MultistageProblem,
    V::Vector{PolyhedralFunction},
    D::Vector{PolyhedralFunction},
    scenario::InSampleScenario{Float64},
    initial_state::Vector{Float64},
)
    T = horizon(tree.model)
    primal_trajectory = fill(0.0, length(initial_state), T + 1)
    return forward_pass!(sddp, tree, V, D, scenario, initial_state, primal_trajectory)
end

function build_tree(solver::RegularizedPrimalSDDP, hdm::HazardDecisionModel, V::Vector{PolyhedralFunction}; mode::Int = 1)
    # Get number of nodes per stage.
    T = horizon(hdm)
    # Build multistage tree
    stages = RegularizedStage{JuMP.Model}[]
    parent = nothing
    for t in 1:T
        # Build default multistage problem.
        npb = stage_model(hdm, t) # original problem
        rpb = stage_model(hdm, t) # regularized problem
        is_final = (t == T)
        stage = RegularizedStage(
            parent, npb, rpb, t, is_final,
        )
        push!(stages, stage)
        parent = stage
    end
    # Initialize JuMP model
    for stage in stages
        if !stage.is_final
            initialize!(solver, stage, V[stage.t+1]; mode = mode)
        end
        JuMP.set_optimizer(stage.model, solver.primal_sddp.optimizer)
        JuMP.set_optimizer(stage.regularized_model, solver.qp_optimizer)
    end
    return MultistageProblem(hdm, stages)
end

function solve!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int=1,
    τ=1e8,
    mode::Int = 1,
    saving_data::Bool=false,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    # Build regularized tree.
    ptree = build_tree(solver, hdm, V; mode = mode)
    # NB: we don't regularize dual SDDP.
    dtree = build_tree(solver.dual_sddp, hdm, D)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %15s %5s\n", "-"^4, "-"^15, "-"^15, "-"^15, "-"^5)
        @printf(" %4s %15s %15s %15s %5s\n", "#it", "LB", "UB", "Gap (%)", "lvl")
    end

    if saving_data
        df = init_save(hdm, n_iter)
    end

    # Run
    tic = time()
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    for i in 1:n_iter
        scenario = sample(Ξ)
        if saving_data
            # Primal
            df.timers[i, :time_primal_forward] = @elapsed( (primal_trajectory, solve_cnt, level_cnt) = forward_pass(solver, ptree, V, D, scenario, x₀) )
            df.timers[i, :time_primal_backward] = @elapsed(dual_trajectory = backward_pass!(solver.primal_sddp, ptree, primal_trajectory, V))
            # Dual 
            df.timers[i, :time_dual_forward] = @elapsed( backward_pass!(solver.dual_sddp, dtree, dual_trajectory, D))
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
            df.timers[i,:time_dual_backward] = @elapsed( forward_pass!(solver.dual_sddp, dtree, scenario, p₀, D))
            for t in 1:horizon(hdm)
                df.lb[i,t+1] = V[t](primal_trajectory[:, t]) 
                df.ub[i, t+1] = fenchel_transform(solver.dual_sddp, D[t], primal_trajectory[:, t])[1]
            end
        else
            # Primal
            primal_trajectory, solve_cnt, level_cnt = forward_pass(solver, ptree, V, D, scenario, x₀)
            dual_trajectory = backward_pass!(solver.primal_sddp, ptree, primal_trajectory, V)
            # Dual
            backward_pass!(solver.dual_sddp, dtree, dual_trajectory, D)
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
            forward_pass!(solver.dual_sddp, dtree, scenario, p₀, D)
        end

        lb = V[1](x₀)
        if (verbose > 0) && (mod(i, verbose) == 0)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %15.3f %5i\n", i, lb, ub, 100 * gap, level_cnt)
        end
    end

    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %15s %5s\n\n", "-"^4, "-"^15, "-"^15, "-"^15, "-"^5)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_data.csv", df.data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_timers.csv", df.timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_lb.csv", df.lb)
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_ub.csv", df.ub) 
    end 
    return (ptree, dtree)
end

# Helper function
function regularizedsddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer_lp,
    optimizer_qp;
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    mode::Int=1,
    saving_data::Bool=false,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer_lp, valid_statuses)
    dual_sddp = DualSDDP(optimizer_lp, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(optimizer_qp, primal_sddp, dual_sddp, τ, mixing)
    primal_models, dual_models = solve!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, verbose=verbose, τ=τ, mode = mode, saving_data = saving_data)

    # Get upper-bound
    ub, _ = fenchel_transform(dual_sddp, D[1], x₀)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀),
        dual_cuts=D,
        dual_models=dual_models,
        upper_bound=ub,
    )
end

function regularizedsddp_givenVD(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer_lp,
    optimizer_qp,
    V::Vector{PolyhedralFunction},
    D::Vector{PolyhedralFunction};
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lip_ub=+1e5,
    lip_lb=-1e5,
    valid_statuses=[MOI.OPTIMAL],
    mode::Int=1,
    saving_data::Bool=false,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)

    # Solvers
    primal_sddp = SDDP(optimizer_lp, valid_statuses)
    dual_sddp = DualSDDP(optimizer_lp, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(optimizer_qp, primal_sddp, dual_sddp, τ, mixing)
    primal_models, dual_models = solve_givenVD!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, verbose=verbose, τ=τ, mode = mode, saving_data = saving_data)

    # Get upper-bound
    ub, _ = fenchel_transform(dual_sddp, D[1], x₀)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀),
        dual_cuts=D,
        dual_models=dual_models,
        upper_bound=ub,
    )
end

function solve_givenVD!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int=1,
    τ=1e8,
    mode::Int = 1,
    saving_data::Bool=false,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    # Build regularized tree.
    ptree = build_tree(solver, hdm, V; mode = mode)
    # NB: we don't regularize dual SDDP.
    dtree = build_tree(solver.dual_sddp, hdm, D)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %15s %5s\n", "-"^4, "-"^15, "-"^15, "-"^15, "-"^5)
        @printf(" %4s %15s %15s %15s %5s\n", "#it", "LB", "UB", "Gap (%)", "lvl")
    end

    if saving_data
        df = init_save(hdm, n_iter)
    end

    # Run
    tic = time()
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    for i in 1:n_iter
        scenario = sample(Ξ)
        if saving_data
            # Primal
            df.timers[i, :time_primal_forward] = @elapsed( (primal_trajectory, solve_cnt, level_cnt) = forward_pass(solver, ptree, V, D, scenario, x₀) )
            df.timers[i, :time_primal_backward]= @elapsed( dual_trajectory = backward_pass!(solver.primal_sddp, ptree, primal_trajectory, V))
            # Dual 
            df.timers[i, :time_dual_backward] = @elapsed( backward_pass!(solver.dual_sddp, dtree, dual_trajectory, D))
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
            # df.timers[i,:time_dual_forward] = @elapsed( forward_pass!(solver.dual_sddp, dtree, scenario, p₀, D))
            for t in 1:horizon(hdm)
                df.lb[i,t+1] = V[t](primal_trajectory[:, t]) 
                df.ub[i, t+1] = fenchel_transform(solver.dual_sddp, D[t], primal_trajectory[:, t])[1]
            end

            for t in 2:horizon(hdm)
                df.Δ_norm[i,t] = norm(primal_trajectory[:,t]-primal_trajectory[:,t-1])
            end 

            if i in [100, 300]
                #save("Vreg_$(i).jld2", Dict("V"=>V))
                if saving_data 
                    CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_data.csv", df.data) 
                    CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_timers.csv", df.timers) 
                    CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_ub.csv", df.ub) 
                    CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_lb.csv", df.lb)
                    CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_deltanorm.csv", df.Δ_norm) 
                end 
            end
        else
            # Primal
            primal_trajectory, solve_cnt, level_cnt = forward_pass(solver, ptree, V, D, scenario, x₀)
            dual_trajectory = backward_pass!(solver.primal_sddp, ptree, primal_trajectory, V)
            # Dual
            backward_pass!(solver.dual_sddp, dtree, dual_trajectory, D)
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
            forward_pass!(solver.dual_sddp, dtree, scenario, p₀, D)
        end

        lb = V[1](x₀)
        if (verbose > 0) && (mod(i, verbose) == 0)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %15.3f %5i\n", i, lb, ub, 100 * gap, level_cnt)
        end
    end

    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %15s %5s\n\n", "-"^4, "-"^15, "-"^15, "-"^15, "-"^5)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_data.csv", df.data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_timers.csv", df.timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_lb.csv", df.lb)
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_ub.csv", df.ub) 
        CSV.write(lowercase(split(name(hdm))[1])*"_regsddp_deltanorm.csv", df.Δ_norm) 
    end 
    return (ptree, dtree)
end
