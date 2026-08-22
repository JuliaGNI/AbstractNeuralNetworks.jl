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
and fills `x` in place. `DefaultInitializer` is [`GlorotUniform`](@ref).

"""
function initialparameters end

initialparameters(rng::AbstractRNG, initializer::Initializer, model::Model, ::NeuralNetworkBackend, ::Type{T}; kwargs...) where T = error("initialparameters not implemented for model type ", typeof(model))

function parameterlength end

Base.eachindex(m::Model) = @error "You forgot to define the eachindex function for the model of type "*string(typeof(m))*"!"

update!(model::Model, params::Union{NamedTuple,NetworkParameters}, grad::Union{NamedTuple,NetworkParameters}, args...) = update!(model, values(params), values(grad), args...)
