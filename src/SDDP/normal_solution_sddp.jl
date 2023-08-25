
#=
    Regularized primal SDDP
=#

struct NormalSolutionSDDP <: AbstractRegularizedSDDP
    qp_optimizer::Any
    primal_sddp::SDDP
    tau::Float64
    mixing::Float64
end

introduce(::NormalSolutionSDDP) = "Normal Solution SDDP"

function build_tree(solver::NormalSolutionSDDP, hdm::HazardDecisionModel, V::Vector{PolyhedralFunction}; 
    mode::Int=1,) 
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

function forward_pass!(
    normsol_sddp::NormalSolutionSDDP,
    tree::MultistageProblem,
    V::Vector{PolyhedralFunction},
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
        lb = lowerbound(normsol_sddp.primal_sddp, stage, xₜ, wₜ)

        # Regularization level to -∞, ensures that we compute the normal solution of stage prob
        ℓ = -1e9
        xₜ, is_level = next!(normsol_sddp, stage, xₜ, Ξ[stage.t], wₜ, ℓ)
        solve_cnt += 1
        level_cnt += is_level

        trajectory[:, stage.t+1] .= xₜ
    end
    return trajectory, solve_cnt, level_cnt
end

function forward_pass(
    sddp::NormalSolutionSDDP,
    tree::MultistageProblem,
    V::Vector{PolyhedralFunction},
    scenario::InSampleScenario{Float64},
    initial_state::Vector{Float64},
)
    T = horizon(tree.model)
    primal_trajectory = fill(0.0, length(initial_state), T + 1)
    return forward_pass!(sddp, tree, V, scenario, initial_state, primal_trajectory)
end

function solve!(
    solver::NormalSolutionSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int=1,
    τ=1e8,
    mode::Int=1,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    # Build regularized tree.
    ptree = build_tree(solver, hdm, V; mode = mode)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %5s\n", "-"^4, "-"^15, "-"^5)
        @printf(" %4s %15s %5s\n", "#it", "LB", "lvl")
    end

    # Run
    tic = time()
    for i in 1:n_iter
        scenario = sample(Ξ)
        # Primal
        primal_trajectory, solve_cnt, level_cnt = forward_pass(solver, ptree, V, scenario, x₀;)
        backward_pass!(solver.primal_sddp, ptree, primal_trajectory, V)

        lb = V[1](x₀)
        if (verbose > 0) && (mod(i, verbose) == 0)
            @printf(" %4i %15.6e %5i\n", i, lb, level_cnt)
        end
    end

    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %15s %5s\n\n", "-"^4, "-"^15, "-"^15, "-"^15, "-"^5)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
    end
    return (ptree)
end

# Helper function
function normalsolutionsddp(
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
    valid_statuses=[MOI.OPTIMAL],
    mode::Int = 1,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer_lp, valid_statuses)

    # Solve
    normsol_sddp = NormalSolutionSDDP(optimizer_qp, primal_sddp, τ, mixing)
    primal_models = solve!(normsol_sddp, hdm, V, x₀; n_iter=n_iter, verbose=verbose, τ=τ, mode = mode)

    return (
        primal_cuts=V,
        primal_models=primal_models,
        lower_bound=V[1](x₀)
    )
end

