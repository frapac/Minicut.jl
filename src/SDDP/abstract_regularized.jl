
#=
    Generic structures for regularized SDDP.
=#

abstract type AbstractRegularizedSDDP <: AbstractSDDP end

abstract type AbstractRegularizedNode <: AbstractNode end

function initialize!(sddp::AbstractRegularizedSDDP, stage::AbstractRegularizedNode, Vₜ₊₁::PolyhedralFunction; mode::Int = 1)
    # Original model
    @variable(stage.model, θ)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(stage.model, θ >= λ' * stage.model[_CURRENT_STATE] + γ)
    end
    obj_expr = JuMP.objective_function(stage.model)
    @objective(stage.model, Min, obj_expr + θ)

    # Regularized model
    @expression(stage.regularized_model, cost, JuMP.objective_function(stage.regularized_model))
    @variable(stage.regularized_model, θ)
    @variable(stage.regularized_model, w)
    @variable(stage.regularized_model, level)
    for (λ, γ) in eachcut(Vₜ₊₁)
        @constraint(stage.regularized_model, θ >= λ' * stage.regularized_model[_CURRENT_STATE] + γ)
    end
    @constraint(stage.regularized_model, w >= θ + cost )
    @constraint(stage.regularized_model, w >= level)
    if mode == 1
        @objective(stage.regularized_model, Min, w + (1 / (2 * sddp.tau)) * sum(stage.regularized_model[_CURRENT_STATE]).^2)
    elseif mode == 2
        @variable(stage.regularized_model, y)
        @objective(stage.regularized_model, Min, w + y)
        t = @expression(stage.regularized_model, 1+y)
        @constraint(stage.regularized_model, [t, vcat(sqrt(2)*sddp.tau^(-0.5)*stage.regularized_model[_CURRENT_STATE], 1 - y)] ∈ SecondOrderCone())
    elseif mode == 3
        @variable(stage.regularized_model, y )

        @objective(stage.regularized_model, Min, w + y)
        nx = length(stage.regularized_model[_CURRENT_STATE])
        @constraint(stage.regularized_model, abs_pos[i=1:nx], stage.regularized_model[_CURRENT_STATE][i] ≤ y)
        @constraint(stage.regularized_model, abs_neg[i=1:nx], -stage.regularized_model[_CURRENT_STATE][i] ≤ y)
    end

    # @constraint(stage.regularized_model, θ + cost >= level)
    # @objective(stage.regularized_model, Min, θ + cost + (1 / (2 * sddp.tau)) * sum(stage.regularized_model[_CURRENT_STATE]).^2)

    return
end

function stage_objective_value(sddp::AbstractRegularizedSDDP, stage::AbstractRegularizedNode)
    if stage.is_final
        return JuMP.objective_value(stage.regularized_model)
    else
        return JuMP.value(stage.regularized_model[:cost])
    end
end

function add_cut!(stage::AbstractRegularizedNode, Vₜ::PolyhedralFunction, λ, γ)
    add_cut!(Vₜ, λ, γ)

    parent = stage.parent
    if !isnothing(parent)
        # Add cut both in original and regularized models.
        for model in [parent.model, parent.regularized_model]
            @constraint(model, model[_VALUE_FUNCTION] >= λ' * model[_CURRENT_STATE] + γ)
        end
    end
    return
end

function next!(
    sddp::AbstractRegularizedSDDP,
    stage::AbstractRegularizedNode,
    xₜ::Vector{Float64},
    ξ::DiscreteRandomVariable{Float64},
    ξₜ₊₁::Vector{Float64},
    level::Float64,
)
    fix.(stage.regularized_model[_PREVIOUS_STATE], xₜ, force = true)
    fix.(stage.regularized_model[_UNCERTAINTIES], ξₜ₊₁, force = true)
    if !stage.is_final
        fix.(stage.regularized_model[_LEVEL], level, force = true)
    end
    optimize!(stage.regularized_model)
    status = termination_status(stage.regularized_model)
    if status ∉ sddp.primal_sddp.valid_statuses
        error("[SDDP] Fail to solve primal subproblem: solver's return status is $(status).")
    end
    if !stage.is_final
        w = JuMP.value(stage.regularized_model[:w])
        is_level = isapprox(w, level)
    else
        is_level = false
    end
    return JuMP.value.(stage.regularized_model[_CURRENT_STATE]), is_level
end

