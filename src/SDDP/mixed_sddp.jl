#=
    Mixed primal-dual SDDP
=#

struct MixedPrimalDualSDDP <: AbstractSDDP
    primal_sddp::SDDP
    dual_sddp::DualSDDP
end

introduce(::MixedPrimalDualSDDP) = "Mixed Primal-Dual SDDP"

function solve!(
    solver::MixedPrimalDualSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    verbose::Int = 1,
    saving_data = false,
    allowed_time = 1200,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)
    dual_models = build_stage_models(solver.dual_sddp, hdm, D)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end

    run_data, run_timers, run_ub, run_lb = init_data(solver,
    primal_models,
    hdm,
    V,
    x₀,
    allowed_time,
    n_iter,
    )

    # Run
    ub = Inf
    tic = time()
    for i in 1:n_iter
        scenario = sample(Ξ)
        # Primal
        tic = time()
        primal_trajectory = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
        run_timers[i, :time_primal_forward] += time() - tic 
        tic = time()
        dual_trajectory = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectory, V)
        run_timers[i, :time_primal_backward] += time() - tic 
        # Dual
        tic = time()
        backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectory, D)
        run_timers[i, :time_dual_backward] += time() - tic 
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        tic = time()
        cupps_pass!(solver.dual_sddp, hdm, dual_models, scenario, p₀, D)
        run_timers[i, :time_dual_forward] += time() - tic 

        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        if saving_data
            for t in 1:horizon(hdm)
                run_ub[i,t+1] = fenchel_transform(solver.dual_sddp, D[t], primal_trajectory[:, t])[1]
                run_lb[i,t+1] = V[t](primal_trajectory[:, t]) 
            end
        end
    end

    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_rundata_mixedsddp.csv", run_data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runtimers_mixedsddp.csv", run_timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runub_mixedsddp.csv", run_ub) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runlb_mixedsddp.csv", run_lb) 
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
function mixedsddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    seed=0,
    n_iter=500,
    verbose::Int = 1,
    lower_bound=-1e6,
    lip_ub=+1e10,
    lip_lb=-1e10,
    valid_statuses=[MOI.OPTIMAL],
    allowed_time = 1200,
    saving_data = false,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    # Polyhedral functions.
    V = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    D = [PolyhedralFunction(nx, lower_bound) for t in 1:T]
    # Solvers.
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    mixed_sddp = MixedPrimalDualSDDP(primal_sddp, dual_sddp)
    primal_models, dual_models = solve!(mixed_sddp, hdm, V, D, x₀; n_iter=n_iter, verbose=verbose, allowed_time = allowed_time, saving_data = saving_data)

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

