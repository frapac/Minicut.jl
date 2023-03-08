#=
    Build dual one-stage model associated to
    a given primal one-stage model.

    Hazard-decision information structure is assumed.

    Transformation occurs in two steps:

    1. The one-stage model is unraveled to an extensive stage model:
                                        ξ₁
       ξ                              /---> xf₁
    x ---> xf        becomes         x  ξ₂
                                      \---> xf₂

    2. The extensive stage model is dualized using Dualization.jl

=#

_get_rhs(set::MOI.LessThan{T}) where {T} = set.upper
_get_rhs(set::MOI.GreaterThan{T}) where {T} = set.lower
_get_rhs(set::MOI.EqualTo{T}) where {T} = set.value

function _get_extensive_stage_problem(model::JuMP.Model, ξ::DiscreteRandomVariable{Float64})
    # Extensive model
    dest = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), MOIU.AUTOMATIC)
    nξ = length(ξ)

    # Get MOI representation
    src = JuMP.backend(model)

    # Indexing
    all_index = MOI.get(src, MOI.ListOfVariableIndices())
    index_x = index.(model[_PREVIOUS_STATE])
    index_xf = index.(model[_CURRENT_STATE])
    index_ξ = index.(model[_UNCERTAINTIES])
    index_edges = [iu for iu in all_index if iu ∉ [index_x; index_ξ]]
    nx, nu = length(index_x), length(index_edges)

    # Load objective functions
    ObjFunc = MOI.get(src, MOI.ObjectiveFunctionType())
    func_obj = MOI.get(src, MOI.ObjectiveFunction{ObjFunc}())
    # Terms for extensive objective
    constant_obj = func_obj.constant
    terms_obj = MOI.ScalarAffineTerm[]

    # TODO: use MOI.IndexMap here
    index_map_root = Dict()
    new_index_x = MOI.VariableIndex[]
    new_index_xf = MOI.VariableIndex[]

    # Build state variable xₜ
    for xi in index_x
        index_map_root[xi] = MOI.add_variable(dest)
        push!(new_index_x, index_map_root[xi])
        name = MOI.get(src, MOI.VariableName(), xi)
        MOI.set(dest, MOI.VariableName(), index_map_root[xi], name)
    end

    for k in 1:nξ
        # Current proba
        w = ξ.weights[k]
        # Correspondance map for state
        index_map = copy(index_map_root)
        for iu in index_edges
            index_map[iu] = MOI.add_variable(dest)
            name = MOI.get(src, MOI.VariableName(), iu)
            MOI.set(dest, MOI.VariableName(), index_map[iu], "$(name)_$k")
            if iu in index_xf
                push!(new_index_xf, index_map[iu])
            end
        end
        # Correspondance map for uncertainties
        for (l, iξ) in enumerate(index_ξ)
            index_map[iξ] = ξ.supports[l, k]
        end
        # Rewrite bound constraints as generic constraints
        # and scale them by probability weights.
        for S in [MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}]
            for con in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
                idx = MOI.get(src, MOI.ConstraintFunction(), con)
                set = MOI.get(src, MOI.ConstraintSet(), con)
                if idx in index_edges
                    rhs = _get_rhs(set)
                    new_set = S(w * rhs)
                    term = MOI.ScalarAffineTerm{Float64}(w, index_map[idx])
                    new_func = MOI.ScalarAffineFunction{Float64}([term], 0.0)
                    con = MOI.add_constraint(dest, new_func, new_set)
                end
            end
        end
        # Adapt generic constraints and add a reference
        # to the coupling constraints between xₜ and xₜ₊₁.
        # All terms are scaled by current probability weights w.
        id_coupling = 0
        id_operational = 0
        for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
            # Skip variable's bounds
            (F <: MOI.VariableIndex) && continue
            for con in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
                func = MOI.get(src, MOI.ConstraintFunction(), con)
                set = MOI.get(src, MOI.ConstraintSet(), con)
                terms = MOI.ScalarAffineTerm[]
                constant = w * func.constant
                rhs = w * _get_rhs(set)
                new_set = S(rhs)
                is_coupling = false
                # Read original constraint
                for t in func.terms
                    vi = index_map[t.variable]
                    if vi isa Float64
                        constant -= w * vi
                    else
                        push!(terms, MOI.ScalarAffineTerm{Float64}(w * t.coefficient, vi))
                    end
                    # Determine if constraint is coupling
                    if vi in index_x
                        is_coupling = true
                    end
                end
                # Build new constraint
                new_func = MOI.ScalarAffineFunction{Float64}(terms, constant)
                con = MOIU.normalize_and_add_constraint(dest, new_func, new_set)
                # Name coupling constraint for future reference
                name = if is_coupling
                    id_coupling += 1
                    "coupling_$(k)_$(id_coupling)"
                else
                    id_operational += 1
                    "op_$(k)_$(id_operational)"
                end
                MOI.set(dest, MOI.ConstraintName(), con, name)
            end
        end
        # Adapt objective
        for t in func_obj.terms
            push!(terms_obj, MOI.ScalarAffineTerm{Float64}(w * t.coefficient, index_map[t.variable]))
        end
    end

    # Objective
    new_obj = MOI.ScalarAffineFunction{Float64}(terms_obj, constant_obj)
    MOI.set(dest, MOI.ObjectiveFunction{ObjFunc}(), new_obj)
    MOI.set(dest, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return dest, new_index_x, new_index_xf
end

# Build dual model using Dualization and add variables
# associated to the costate μₜ in the model.
function _build_dual_model(primal_model, index_x, index_xf, ξ, lip_lb, lip_ub)
    nx = length(index_x)
    nw = length(ξ)
    dual = Dualization.dualize(primal_model; dual_names = DualNames("", ""))
    # Add adjoint variable μₜ
    index_mu = MOI.VariableIndex[]
    for (i, xi) in enumerate(index_x)
        con = dual.primal_dual_map.primal_var_dual_con[xi]
        vi = MOI.add_variable(dual.dual_model)
        MOI.set(dual.dual_model, MOI.VariableName(), vi, "μ$i")
        # TODO: check sign
        MOI.modify(dual.dual_model, con, MOI.ScalarCoefficientChange(vi, 1.0))
        push!(index_mu, vi)
    end

    # Add adjoint variable μₜ₊₁
    index_mu_next = MOI.VariableIndex[]
    idx = 1
    for k in 1:nw, i in 1:nx
        w = ξ.weights[k]
        xf = index_xf[idx]
        vi = MOI.add_variable(dual.dual_model)
        MOI.set(dual.dual_model, MOI.VariableName(), vi, "μ$(i)_$(k)")
        push!(index_mu_next, vi)
        MOI.add_constraint(dual.dual_model, vi, MOI.LessThan{Float64}(lip_ub))
        MOI.add_constraint(dual.dual_model, vi, MOI.GreaterThan{Float64}(lip_lb))

        # Modify coupling constraints
        con = dual.primal_dual_map.primal_var_dual_con[xf]
        # TODO: check sign
        MOI.modify(dual.dual_model, con, MOI.ScalarCoefficientChange(vi, -w))

        idx += 1
    end

    return dual, index_mu, index_mu_next
end

function dual_stage_model(hd::HazardDecisionModel, t::Int, lip_lb::Float64, lip_ub::Float64)
    Ξ = uncertainties(hd)
    model = stage_model(hd, t)

    # Build extensive one-stage model
    dest, index_x, index_xf = _get_extensive_stage_problem(model, Ξ[t])
    # Build dual MOI model
    dual, index_mu, index_mu_next = _build_dual_model(dest, index_x, index_xf, Ξ[t], lip_lb, lip_ub)

    # Copy dual MOI model into a JuMP model
    dual_model = JuMP.Model()
    idx_map = MOI.copy_to(dual_model, dual.dual_model)
    # Build correspondance for dual co-states
    dual_model[_PREVIOUS_COSTATE] = JuMP.VariableRef[JuMP.VariableRef(dual_model, idx_map[ix]) for ix in index_mu]
    dual_model[_CURRENT_COSTATE] = JuMP.VariableRef[JuMP.VariableRef(dual_model, idx_map[ix]) for ix in index_mu_next]

    return dual_model
end
