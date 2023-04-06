
function warmup!(
    solver::RegularizedPrimalSDDP,
    primal_models,
    dual_models,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    verbose::Int=1,
    n_warming = 50
)
    Ξ = uncertainties(hdm)
    ub = Inf
    for i in 1:n_warming
        scenario = sample(Ξ)
        # Primal
        primal_trajectory = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
        dual_trajectory = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectory, V)
        # Dual
        backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectory, D)
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)

        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
    end
end


function solve2!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    n_cycle = 20,
    verbose::Int=1,
    τ=1e8,
    n_prunning = 100,
    allowed_time = 300,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)
    dual_models = build_stage_models(solver.dual_sddp, hdm, D)

    # Warmup
    tic = time()
    warmup!(solver, primal_models, dual_models, hdm, V, D, x₀; verbose=verbose, n_warming = 50)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    # Run
    ub = Inf
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    primal_trajectories = Array{Matrix{Float64}}(undef, n_prunning)
    dual_trajectories = Array{Matrix{Float64}}(undef, n_prunning)
    for i in 1:n_iter
        scenario = sample(Ξ)
        j = mod(i, n_prunning) + 1 # Current index since last prunning
        if mod(i, n_cycle) == 0 # Update upper bounds via dual sddp
            # Primal
            primal_trajectories[j] = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
            backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            # Dual
            dual_trajectories[j] = forward_pass(solver.dual_sddp, hdm, dual_models, scenario, p₀)
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        else # Update upper bounds using primal cuts = dual trajectory
            # Primal
            primal_trajectories[j] = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
            dual_trajectories[j] = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            # Dual
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        end
        if  j == n_prunning
            V_ref = V 
            D_ref = D
            V = [PolyhedralFunction(length(x₀), V[1](x₀)) for t in 1:length(V)]
            D = [PolyhedralFunction(length(x₀), D[1](dual_trajectories[j][:, 1])) for t in 1:length(D)]
            for j in length(primal_trajectories)
                prunning!(V, V_ref, primal_trajectories[j])
                prunning!(D, D_ref, dual_trajectories[j])
            end
        end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        # Check if allowed time is over 
        if time() - tic > allowed_time
            break
        end
    end

    status = MOI.ITERATION_LIMIT
    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %10s\n\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    return (primal_models, dual_models)
end

# Helper function
function regularizedsddp2(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    mixing=1.0,
    τ=1e8,
    seed=0,
    n_iter=500,
    verbose::Int=1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    n_cycle= 10,
    n_prunning = 100,
    allowed_time = 300,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Primal Polyhedral function
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]

    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)
    
    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing)
    primal_models, dual_models = solve2!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, n_cycle=n_cycle, verbose=verbose, τ=τ, n_prunning = n_prunning, allowed_time=allowed_time)

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

