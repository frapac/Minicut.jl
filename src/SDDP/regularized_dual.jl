#=
    Regularized dual SDDP
=#

struct RegularizedDualSDDP <: AbstractSDDP
    primal_sddp::SDDP
    dual_sddp::DualSDDP
    tau::Float64
    mixing::Float64
    name::String
end 

introduce(regsddp::RegularizedPrimalSDDP) = regsddp.name

function solve2!(
    solver::RegularizedDualSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    allowed_time = 300,
    n_cycle = 20,
    n_iter=100,
    n_pruning = 100,
    n_warmup = 50,
    τ=1e8,
    verbose::Int=1,
    saving_data=false,
)
    (verbose > 0) && header()
    Ξ = uncertainties(hdm)

    primal_models = build_stage_models(solver.primal_sddp, hdm, V)
    dual_models = build_stage_models(solver.dual_sddp, hdm, D)

    # Saving run data in a DataFrame pb_data, df_timers, df_ub, df_lb, df_traj
    run_data, run_timers, run_ub, run_lb = init_data(solver.primal_sddp,
    primal_models,
    hdm,
    V,
    x₀,
    allowed_time,
    n_cycle,
    n_iter,
    n_pruning,
    n_warmup,
    )
    # Warmup
    
    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
    end

    if n_warmup > 0
        println("Warming up")
        V, D = warmup!(solver, primal_models, dual_models, hdm, V, D, x₀; verbose=verbose, n_warmup = n_warmup, n_pruning = n_pruning)
    end 

    if verbose > 0
        @printf(" %4s %15s %15s %10s\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf(" %4s %15s %15s %10s\n", "#it", "LB", "UB", "Gap (%)")
    end
    
    # Timers
    time_mainrun = time()

    # Run
    ub = Inf
    ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
    primal_trajectories = Array{Matrix{Float64}}(undef, n_pruning)
    #primal_trajectories_test = Array{Matrix{Float64}}(undef, n_pruning)
    dual_trajectories = Array{Matrix{Float64}}(undef, n_pruning)

    for i in 1:n_iter
        tic_iter = time()
        scenario = sample(Ξ)
        j = mod(i, n_pruning) + 1 # Current index since last pruning
        if mod(i, n_cycle) == 0 # Update upper bounds via dual sddp
            # Dual
            tic = time()
            dual_trajectories[j] = forward_pass(solver.dual_sddp, hdm, dual_models, scenario, p₀)
            run_timers[i, :time_dual_forward] += time() - tic 
            tic = time()
            primal_trajectories[j] = backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
            run_timers[i, :time_dual_backward] += time() - tic 
            # Primal
            tic = time()
            #primal_trajectories[j], ub_tmp = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
            #primal_trajectories[j] = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
            run_timers[i, :time_primal_forward] += time() - tic 
            tic = time()
            backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            run_timers[i, :time_primal_backward] += time() - tic 

        else # Update upper bounds using primal cuts = dual trajectory
            # Primal
            tic = time()
            #primal_trajectories[j], ub_tmp = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
            primal_trajectories[j] = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
            run_timers[i, :time_primal_forward] += time() - tic 
            tic = time()
            dual_trajectories[j] = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            run_timers[i, :time_primal_backward] += time() - tic 
            run_timers[i, :time_dual_forward] += 0.0 
            # Dual
            tic = time()
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
            run_timers[i, :time_dual_backward] += time() - tic
        end
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        if  (n_pruning != 1) && (j == 1)
            tic = time()
            V = pruning(V, primal_trajectories; verbose = verbose)
            #D = pruning(D, dual_trajectories; verbose = verbose)
            run_timers[i, :time_pruning] += time() - tic 
        end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        run_timers[i, :time_iter] += time() - tic_iter
        
        if saving_data
            #run_ub[i, 2:(horizon(hdm)+1)] = ub_tmp
            for t in 1:horizon(hdm)
                run_ub[i,t+1] = fenchel_transform(solver.dual_sddp, D[t], primal_trajectories[j][:, t])[1]
                run_lb[i,t+1] = V[t](primal_trajectories[j][:, t]) # not lb, just values along trajectories
                if run_ub[i, t+1] < run_lb[i,t+1] 
                    println("The upper bound at time $t iteration $i is under the lower bound")
                end
            end
        end
        # Check if allowed time is over
        if time() - time_mainrun > allowed_time
            break
        end
    end

    status = MOI.ITERATION_LIMIT
    # Final status
    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s %15s %10s\n\n", "-"^4, "-"^15, "-"^15, "-"^10)
        @printf("Max number of iterations.........: %7i\n", n_iter)
        @printf("Main loop wall-clock time (sec)..: %7.3f\n\n", time() - time_mainrun)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    #run_data, run_timers, run_ub, run_lb, run_traj
    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_rundata_regsddp.csv", run_data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runtimers_regsddp.csv", run_timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runub_regsddp.csv", run_ub) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runlb_regsddp.csv", run_lb) 
    end 
    return (primal_models, dual_models)
end