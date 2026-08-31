"""
A supertype for `Chain` and the layer types.
"""
abstract type Model end

"""
    initialparameters

Returns the initial parameters of a model, i.e., a layer or chain.

```
initialparameters(rng::AbstractRNG, init::Initializer, model::Model, backend::NeuralNetworkBackend, ::Type{T}; kwargs...)
```

An [`Initializer`](@ref) is called as
```
init(rng::AbstractRNG, x::AbstractArray)
```
and fills `x` in place. [`DefaultInitializer`](@ref) is [`GlorotUniform`](@ref).

A model whose parameters are to be stored in a [`NeuralNetwork`](@ref) must return a
`NetworkParameters`, as [`Chain`](@ref) does. A layer returns the plain `NamedTuple` that its own
functor takes.

"""
function initialparameters end

function initialparameters(rng::AbstractRNG, initializer::Initializer, model::Model,
        ::NeuralNetworkBackend, ::Type{T}; kwargs...) where {T}
    error("initialparameters not implemented for model type ", typeof(model))
end

function parameterlength end

function Base.eachindex(m::Model)
    @error "You forgot to define the eachindex function for the model of type "*string(typeof(m))*"!"
end

function update!(model::Model, params::NetworkParameters, grad::NetworkParameters, args...)
    update!(model, values(params), values(grad), args...)
end
