"""
A supertype for `Chain`, `AbstractCell` etc.
"""
abstract type Model end


"""
    initialparameters

    Returns the initial parameters of a model, i.e., a layer or chain.

```
initialparameters(backend::Backend, ::Type{T}, model::Model; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng())
initialparameters(::Type{T}, model::Model; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng())
```

The `init!` function must have the following signature:
```
init!(rng::AbstractRNG, x::AbstractArray)
```
The `default_initializer()` returns `randn!`.

"""
function initialparameters end

initialparameters(rng::AbstractRNG, initializer::Initializer, model::Model, ::Backend, ::Type{T}; kwargs...) where T = error("initialparameters not implemented for model type ", typeof(model))

function parameterlength end

Base.eachindex(m::Model) = @error "You forgot to define the eachindex function for the model of type "*string(typeof(m))*"!"

update!(model::Model, params::Union{NamedTuple,NeuralNetworkParameters}, grad::Union{NamedTuple,NeuralNetworkParameters}, args...) = update!(model, values(params), values(grad), args...)
