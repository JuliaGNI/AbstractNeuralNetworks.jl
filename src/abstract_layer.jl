@doc raw"""
    AbstractLayer

Abstract supertype for all layers.

Layer types should implement the following functions:

- `initialparameters(layer::AbstractLayer, arrtype::Type, initializer::Callable, rng::AbstractRNG)`

and the functors

- `layer(x, ps)`
- `layer(y, x, ps)`

"""
abstract type AbstractLayer end


"""
    apply(layer::AbstractLayer, x, ps)

Simply calls `layer(x, ps)`
"""
function apply(layer::AbstractLayer, x, ps)
    return layer(x, ps)
end

"""
    apply!(y, layer::AbstractLayer, x, ps)

Simply calls `layer(y, x, ps)`
"""
function apply!(::AbstractVector, layer::AbstractLayer, x, ps)
    return layer(y, x, ps)
end


initialparameters(::AbstractRNG, layer::AbstractLayer, ::Callable) = error("initialparameters not implemented for layer type ", typeof(layer))
initialparameters(layer::AbstractLayer, init::Callable = default_initializer()) = initialparameters(Random.default_rng(), layer, init)


"""
    AbstractExplicitLayer

Abstract supertype for explicit layers.
This type exists mainly for compatibility with Lux.
"""
abstract type AbstractExplicitLayer <: AbstractLayer end
