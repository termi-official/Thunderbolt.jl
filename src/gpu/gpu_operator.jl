# TODO: Type stability

###################################
# GPU dispatch for LinearOperator #
###################################

# interfaces #
# Concrete implementations for the following abstract types can be found in the corresponding extension.
abstract type AbstractOperatorKernel{BKD} end

function init_linear_operator(::AbstractAssemblyStrategy,::IntegrandType,::QuadratureRuleCollection,::AbstractDofHandler ) where {IntegrandType}
    error("Not implemented")
end

function init_linear_operator(::AbstractAssemblyStrategy, ::LinearOperator) 
    error("Not implemented")
end

function update_operator!(::AbstractOperatorKernel, time)
    error("Not implemented")
end

