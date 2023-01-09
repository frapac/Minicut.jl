
abstract type AbstractBellmanModel end

abstract type HereAndNowModel <: AbstractBellmanModel end

function stage_model end

function uncertainties end

function horizon end

function number_states end


abstract type AbstractStochasticOptimizer end
