#=
    Solve stochastic problem using extensive formulation.

    The scenario tree is given as a Vector{Node}, storing all the nodes
    in the tree.

    We use MOI to build the JuMP model associated with the extensive form,
    using the function stage_model to build the model at each time t.
=#

struct ExtensiveFormulationSolver <: AbstractStochasticOptimizer
    optimizer::Any
end

#=
    Tree structure
=#

struct Node{T}
    # Parent node. Is nothing for root node.
    parent::Union{Nothing, Node}
    # Correspondance map between model at time t and extensive model
    index_map::Dict
    # Index of state variable attached to this node
    state::Vector{MOI.VariableIndex}
    # Node's probability
    proba::T
    # Time step
    t::Int
end

"""

Add children recursively to the scenario tree.
Update the destination model `dest` encoding the extensive form accordingly.

## Arguments

- `dest::MOI.AbstractOptimizer`: model storing the extensive form.
- `tree::Vector{Node}`: scenario tree
- `parent::Node`: current parent node
- `models::Vector{JuMP.Model}`: one stage model for each time `t`
- `Ξ::Vector{DiscreteRandomVariable}`: uncertainty model
- `final_time::Int`: max-depth of the scenario tree

"""
function add_children!(
    dest::MOI.AbstractOptimizer,
    tree::Vector{Node{T}},
    parent::Node{T},
    models::Vector{JuMP.Model},
    Ξ::Vector{DiscreteRandomVariable{T}},
    final_time::Int,
) where T
    if parent.t == final_time
        return
    end
    # Current time
    t = parent.t + 1
    model = models[t]
    ξ = Ξ[t]

    # Load MOI model
    src = JuMP.backend(model)
    all_index = MOI.get(src, MOI.ListOfVariableIndices())
    index_x = index.(model[_PREVIOUS_STATE])
    index_xf = index.(model[_CURRENT_STATE])
    index_ξ = index.(model[_UNCERTAINTIES])
    index_edges = [iu for iu in all_index if iu ∉ [index_x; index_xf; index_ξ]]
    nx = length(index_x)

    # Load objective src
    ObjFunc = MOI.get(src, MOI.ObjectiveFunctionType())
    obj_src = MOI.get(src, MOI.ObjectiveFunction{ObjFunc}())
    # Load objective dest
    obj_dest = MOI.get(dest, MOI.ObjectiveFunction{ObjFunc}())

    # Get current number of nodes at time t
    nt = sum([1 for node in tree if node.t == t])

    # Build leafs
    for k in 1:length(ξ)
        # Transition proba
        w = ξ.weights[k]
        # Add new state associated to current transition
        ix = MOI.add_variables(dest, nx)
        # Add new leaf
        node = Node(parent, Dict(), ix, w * parent.proba, t)
        # Correspondance between previous state and state in src
        for (iold, inew) in zip(index_x, node.parent.state)
            node.index_map[iold] = inew
        end
        # Correspondance between current state and next state in src
        for (iold, inew) in zip(index_xf, ix)
            node.index_map[iold] = inew
            name = MOI.get(src, MOI.VariableName(), iold)
            MOI.set(dest, MOI.VariableName(), inew, "$(name)_$(t)_$(k+nt)")
        end
        # Correspondance between current controls and controls in src
        for iu in index_edges
            node.index_map[iu] = MOI.add_variable(dest)
            name = MOI.get(src, MOI.VariableName(), iu)
            MOI.set(dest, MOI.VariableName(), node.index_map[iu], "$(name)_$(t)_$(k+nt)")
        end
        # Correspondance for current realization of uncertainty
        for (l, iξ) in enumerate(index_ξ)
            node.index_map[iξ] = ξ.supports[l, k]
        end
        # Copy variable bounds
        for S in [MOI.LessThan{T}, MOI.GreaterThan{T}]
            for con in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, S}())
                idx = MOI.get(src, MOI.ConstraintFunction(), con)
                set = MOI.get(src, MOI.ConstraintSet(), con)
                if idx in [index_edges; index_xf]
                    MOI.add_constraint(dest, node.index_map[idx], set)
                end
            end
        end
        # Adapt edge constraints
        for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
            # Skip variable's bounds
            (F <: MOI.VariableIndex) && continue
            for con in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
                func = MOI.get(src, MOI.ConstraintFunction(), con)
                set = MOI.get(src, MOI.ConstraintSet(), con)
                # Build new constraint in dest
                terms = MOI.ScalarAffineTerm[]
                constant = func.constant
                for t in func.terms
                    vi = node.index_map[t.variable]
                    if vi isa T
                        constant -= vi
                    else
                        push!(terms, MOI.ScalarAffineTerm{T}(t.coefficient, vi))
                    end
                end
                new_func = MOI.ScalarAffineFunction{T}(terms, constant)
                MOIU.normalize_and_add_constraint(dest, new_func, set)
            end
        end
        # Adapt objective
        for t in obj_src.terms
            push!(obj_dest.terms, MOI.ScalarAffineTerm{T}(node.proba * t.coefficient, node.index_map[t.variable]))
        end
        obj_dest.constant += node.proba * obj_src.constant
        # Add new node in tree
        push!(tree, node)
        # Recursion
        add_children!(dest, tree, node, models, Ξ, final_time)
    end
end

function build_scenario_tree(
    hm::HazardDecisionModel;
    max_nodes=1000,
)
    Ξ = Minicut.uncertainties(hm)
    final_time = Minicut.horizon(hm)
    # Test that scenario tree is not too large
    @assert number_nodes(Ξ)[final_time+1] <= max_nodes
    # Load and cache one stage models
    models = [Minicut.stage_model(hm, t) for t in 1:final_time]

    # Scenario tree
    tree = Node{Float64}[]
    # Extensive model
    dest = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        MOIU.AUTOMATIC,
    )
    # Build root node
    nx = Minicut.number_states(hm)
    x0 = MOI.add_variables(dest, nx)
    for (k, ix) in enumerate(x0)
        MOI.set(dest, MOI.VariableName(), ix, "x0[$k]")
    end
    root = Node{Float64}(nothing, Dict(), x0, 1.0, 0)
    push!(tree, root)

    # Build tree recursively
    add_children!(dest, tree, root, models, Ξ, final_time)

    # Set objective properly
    ObjFunc = MOI.ScalarAffineFunction{Float64}
    # The objective has been updated inplace when building the tree
    obj_dest = MOI.get(dest, MOI.ObjectiveFunction{ObjFunc}())
    MOIU.canonicalize!(obj_dest)
    MOI.set(dest, MOI.ObjectiveFunction{ObjFunc}(), obj_dest)
    MOI.set(dest, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return (moi_model=dest, scenario_tree=tree)
end

function solve!(
    ext::ExtensiveFormulationSolver,
    hm::HazardDecisionModel,
    x0::Array;
    max_nodes=5000,
)
    extensive = build_scenario_tree(hm; max_nodes=max_nodes)
    # Copy MOI model to a new JuMP model
    model = JuMP.Model()
    idx = MOI.copy_to(model, extensive.moi_model)
    # Add reference to initial state
    root_node = extensive.scenario_tree[1]
    @assert isnothing(root_node.parent)
    index_x0 = root_node.state
    model[_INITIAL_STATE] = JuMP.VariableRef[JuMP.VariableRef(model, idx[ix]) for ix in index_x0]
    JuMP.fix.(model[_INITIAL_STATE], x0)
    # Solve extensive formulation
    JuMP.set_optimizer(model, ext.optimizer)
    JuMP.optimize!(model)
    return model
end

function extensive(
    hdm::HazardDecisionModel,
    x₀::Array,
    optimizer;
    max_nodes=5000,
)
    extensive_solver = ExtensiveFormulationSolver(optimizer)
    model = solve!(extensive_solver, hdm, x₀; max_nodes=max_nodes)
    return (
        status=JuMP.termination_status(model),
        optimum=JuMP.objective_value(model),
        model=model,
    )
end

