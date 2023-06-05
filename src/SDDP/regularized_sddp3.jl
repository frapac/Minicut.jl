
function solve3!(
    solver::RegularizedPrimalSDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    allowed_time = 300,
    n_cycle = 20,
    n_batch = 10,
    n_iter=100,
    n_pruning = 100,
    n_warmup = 50,
    τ=1e8,
    verbose::Int=1,
    saving_data= true
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
    primal_trajectories = Array{Matrix{Float64}}(undef, n_batch)
    dual_trajectories = Array{Matrix{Float64}}(undef, n_batch)
    ubs = Inf*ones(Float64, length(Ξ), horizon(hdm))

    for i in 1:n_iter
        tic_iter = time()
        scenarios = sample(Ξ, n_batch)
        # Compute updated upperbounds if needed 
            upperbounds!(ubs, Ξ, )
        # Compute forward passes with updated upperbounds
        for k in 1:n_batch
            primal_trajectories[k] = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenarios[k], x₀, τ)
        end
        # Compute backward passes
        dual_trajectories = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories, V)
        if mod(i,n_cycle) == 1 
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories, D)
        end







        if mod(i, n_cycle) == 1 # Update upper bounds via dual sddp
            # Primal
            tic = time()
            primal_trajectories[j] = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
            run_timers[i, :time_primal_forward] += time() - tic 
            tic = time()
            backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            run_timers[i, :time_primal_backward] += time() - tic 
            # Dual
            tic = time()
            dual_trajectories[j] = forward_pass(solver.dual_sddp, hdm, dual_models, scenario, p₀)
            run_timers[i, :time_dual_forward] += time() - tic 
            tic = time()
            backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories, D)
            run_timers[i, :time_dual_backward] += time() - tic 
            ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)

        else # Update only primal, save the cuts as future dual trajectories
            # Primal
            tic = time()
            primal_trajectories[j] = reg_forward_pass(solver, hdm, primal_models, dual_models, V, D, scenario, x₀, τ)
            run_timers[i, :time_primal_forward] += time() - tic 
            tic = time()
            dual_trajectories[j] = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
            run_timers[i, :time_primal_backward] += time() - tic 
            run_timers[i, :time_dual_forward] += 0.0 
            
        end
        if  j == 1
            tic = time()
            V = pruning(V, primal_trajectories; verbose = verbose)
            run_timers[i, :time_pruning] += time() - tic 
        end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
        run_timers[i, :time_iter] += time() - tic_iter
        
        if saving_data
            for t in 1:horizon(hdm)
                # run_ub[i,t+1] = fenchel_transform(solver.dual_sddp, D[t], dual_trajectories[j][:, t])[1]
                run_ub[i,t+1] = fenchel_transform(solver.dual_sddp, D[t], primal_trajectories[j][:, t])[1]
                run_lb[i,t+1] = V[t](primal_trajectories[j][:, t]) # not lb, just values along trajectories
            end
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
        @printf("Max number of iterations.........: %7i\n", n_iter)
        @printf("Main loop wall-clock time (sec)..: %7.3f\n\n", time() - time_mainrun)
        @printf("Lower-bound.....: %15.8e\n", lb)
        @printf("Upper-bound.....: %15.8e\n", ub)
        @printf("Final Gap.......: %13.5f %%\n", 100.0 * (ub - lb) / abs(lb))
    end
    #run_data, run_timers, run_ub, run_lb, run_traj
    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_rundata.csv", run_data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runtimers.csv", run_timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runub.csv", run_ub) 
        CSV.write(lowercase(split(name(hdm))[1])*"_runlb.csv", run_lb) 
    end 
    return (primal_models, dual_models)
end

# Helper function
function regularizedsddp3(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction};
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
    n_pruning = 100,
    allowed_time = 300,
    n_warmup = 50,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)


    # Solvers
    primal_sddp = SDDP(optimizer, valid_statuses)
    dual_sddp = DualSDDP(optimizer, valid_statuses, lip_lb, lip_ub)

    # Solve
    reg_sddp = RegularizedPrimalSDDP(primal_sddp, dual_sddp, τ, mixing, "Regularized Primal SDDP")
    primal_models, dual_models = 3!(reg_sddp, hdm, V, D, x₀; n_iter=n_iter, n_cycle=n_cycle, verbose=verbose, τ=τ, n_pruning = n_pruning, allowed_time=allowed_time, n_warmup = n_warmup)

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


function warmup!(
    solver::RegularizedPrimalSDDP,
    primal_models,
    dual_models,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    D::Array{PolyhedralFunction},
    x₀::Array;
    verbose::Int=1,
    n_warmup = 50,
    n_pruning = 100
)
    Ξ = uncertainties(hdm)
    ub = Inf
    primal_trajectories = Array{Matrix{Float64}}(undef, n_pruning)
    dual_trajectories = Array{Matrix{Float64}}(undef, n_pruning)
    for i in 1:n_warmup
        scenario = sample(Ξ)
        j = mod(i, n_pruning) + 1
        # Primal
        primal_trajectories[j] = forward_pass(solver.primal_sddp, hdm, primal_models, scenario, x₀)
        dual_trajectories[j] = backward_pass!(solver.primal_sddp, hdm, primal_models, primal_trajectories[j], V)
        # Dual
        backward_pass!(solver.dual_sddp, hdm, dual_models, dual_trajectories[j], D)
        ub, p₀ = fenchel_transform(solver.dual_sddp, D[1], x₀)
        if  j == 1
            V = pruning(V,  primal_trajectories, verbose = verbose)
            #D = pruning(D, dual_trajectories, verbose = verbose) # Don't prune the dual anymore as it deteriorates primal upper bounds
        end
        if (verbose > 0) && (mod(i, verbose) == 0) && (n_warmup > 0)
            lb = V[1](x₀)
            gap = (ub - lb) / abs(lb)
            @printf(" %4i %15.6e %15.6e %10.3f\n", i, lb, ub, 100 * gap)
        end
    end
    if n_warmup > 0
        println("Mean primal bundle size after warmup: $(mean(ncuts(V[t]) for t in 1:horizon(hdm)))")
        println("Mean dual bundle size after warmup  : $(mean(ncuts(D[t]) for t in 1:horizon(hdm)))")
    end 
    return V, D
end