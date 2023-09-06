#=
    SDDP
=#

struct SDDP <: AbstractSDDP
    optimizer::Any
    valid_statuses::Vector{MOI.TerminationStatusCode}
end

SDDP(optimizer) = SDDP(optimizer, [MOI.OPTIMAL])

introduce(::SDDP) = "Primal SDDP"

#=
    One-stage problem
=#

function initialize!(::SDDP, stage::AbstractNode, Vₜ₊₁::PolyhedralFunction)
    @variable(stage.model, θ)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(stage.model, θ >= λ' * stage.model[_CURRENT_STATE] + γ)
    end
    obj_expr = objective_function(stage.model)
    @objective(stage.model, Min, obj_expr + θ)
    return
end

function solve_stage_problem!(sddp::SDDP, stage::AbstractNode, xₜ::Vector{Float64}, ξₜ₊₁::Vector{Float64})
    fix.(stage.model[_PREVIOUS_STATE], xₜ, force = true)
    fix.(stage.model[_UNCERTAINTIES], ξₜ₊₁, force = true)
    optimize!(stage.model)
    status = termination_status(stage.model)
    if status ∉ sddp.valid_statuses
        error("[SDDP] Fail to solve primal subproblem: solver's return status is $(status).")
    end
    return
end

fetch_cut(sddp::SDDP, model::JuMP.Model) = dual.(FixRef.(model[_PREVIOUS_STATE]))

function stage_objective_value(sddp::SDDP, stage::AbstractNode, hdm::HazardDecisionModel, t)
    if t == horizon(hdm)
        return JuMP.objective_value(stage.model)
    else
        Vx = JuMP.value.(stage.model[_VALUE_FUNCTION])
        return JuMP.objective_value(stage.model) - Vx
    end
end

function add_cut!(stage::AbstractNode, Vₜ::PolyhedralFunction, λ, γ)
    add_cut!(Vₜ, λ, γ)

    parent = stage.parent
    if !isnothing(parent)
        @constraint(parent.model, parent.model[_VALUE_FUNCTION] >= λ' * parent.model[_CURRENT_STATE] + γ)
    end
    return
end

function next!(
    sddp::SDDP,
    stage::AbstractNode,
    xₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
)
    solve_stage_problem!(sddp, stage, xₜ, ξₜ₊₁)
    return JuMP.value.(stage.model[_CURRENT_STATE])
end

function previous!(
    sddp::SDDP,
    stage::AbstractNode,
    xₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    Vₜ::PolyhedralFunction,
)
    nx = length(xₜ)
    πₜ₊₁, ξₜ₊₁ = ξ.weights, ξ.supports
    λ = zeros(nx)
    γ = 0.0
    for (i, πᵢ) in enumerate(πₜ₊₁)
        solve_stage_problem!(sddp, stage, xₜ, ξₜ₊₁[:, i])
        λᵢ = fetch_cut(sddp, stage.model)
        axpy!(πᵢ, λᵢ, λ)
        γ += πᵢ * (objective_value(stage.model) - dot(λᵢ, xₜ))
    end
    add_cut!(stage, Vₜ, λ, γ)
    return λ
end

function build_tree(solver::SDDP, hdm::HazardDecisionModel, V::Vector{PolyhedralFunction})
    tree = MultistageProblem(hdm)
    for stage in tree.stages
        if stage.t < horizon(hdm)
            initialize!(solver, stage, V[stage.t+1])
        end
        JuMP.set_optimizer(stage.model, solver.optimizer)
    end
    return tree
end

function solve!(
    solver::SDDP,
    hdm::HazardDecisionModel,
    V::Array{PolyhedralFunction},
    x₀::Array;
    n_iter=100,
    n_forward=1,
    verbose::Int = 1,
    saving_data::Bool=false,
)
    (verbose > 0) && header()

    problem = build_tree(solver, hdm, V)
    Ξ = uncertainties(hdm)

    if verbose > 0
        println("Algorithm: ", introduce(solver))
        @printf("\n")
        println(hdm)
        @printf("\n")
        @printf(" %4s %15s\n", "-"^4, "-"^15)
        @printf(" %4s %15s\n", "#it", "LB")
    end
    if saving_data
        df = init_save(hdm, n_iter)
    end

    tic = time()
    # Run
    for i in 1:n_iter
        scen = sample(Ξ, n_forward)
        if saving_data
            df.timers[i, :time_primal_forward] = @elapsed(primal_trajectory = forward_pass(solver, problem, scen, x₀))
            df.timers[i, :time_primal_backward] = @elapsed(dual_trajectory = backward_pass!(solver, problem, primal_trajectory, V))
            for t in 1:horizon(hdm)
                df.lb[i,t+1] = V[t](primal_trajectory[1][:, t]) 
            end
            if i in [100,200,300, n_iter]
                save("V_$(i).jld2", Dict("V"=>V))
            end
        else 
            primal_trajectory = forward_pass(solver, problem, scen, x₀)
            dual_trajectory = backward_pass!(solver, problem, primal_trajectory, V)
        end
        if (verbose > 0) && (mod(i, verbose) == 0)
            lb = V[1](x₀)
            @printf(" %4i %15.6e\n", i, lb)
        end
    end

    if verbose > 0
        lb = V[1](x₀)
        @printf(" %4s %15s\n\n", "-"^4, "-"^15)
        @printf("Number of iterations.........: %7i\n", n_iter)
        @printf("Total wall-clock time (sec)..: %7.3f\n\n", time() - tic)
        @printf("Lower-bound.....: %15.8e\n", lb)
    end

    if saving_data 
        CSV.write(lowercase(split(name(hdm))[1])*"_sddp_data.csv", df.data) 
        CSV.write(lowercase(split(name(hdm))[1])*"_sddp_timers.csv", df.timers) 
        CSV.write(lowercase(split(name(hdm))[1])*"_sddp_lb.csv", df.lb) 
    end 

    return problem
end

# Helper function
function sddp(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    seed=0,
    n_iter=500,
    verbose::Int = 1,
    lower_bound=-1e6,
    valid_statuses=[MOI.OPTIMAL],
    saving_data::Bool=false,
)
    (seed >= 0) && Random.seed!(seed)
    nx, T = number_states(hdm), horizon(hdm)
    V = [PolyhedralFunction(zeros(1, nx), [lower_bound]) for t in 1:T]
    solver = SDDP(optimizer, valid_statuses)
    models = solve!(solver, hdm, V, x₀; n_iter=n_iter, verbose=verbose, saving_data=saving_data)
    return (cuts=V, models=models, lower_bound=V[1](x₀))
end

