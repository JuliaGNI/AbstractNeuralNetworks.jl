@doc raw"""
    AbstractLayer

An `AbstractLayer` is a map from $\mathbb{R}^{M} \rightarrow \mathbb{R}^{N}$.

Concrete layer types should implement the following functions:

- `initialparameters(rng::AbstractRNG, init::Initializer, layer::AbstractLayer, backend::NeuralNetworkBackend, ::Type{T})`
- `update!(::AbstractLayer, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)`

and the functors

- `layer(x, ps)`
- `layer(y, x, ps)`

"""
abstract type AbstractLayer{N,M} <: Model end

input_dimension(::AbstractLayer{M}) where {M} = M
output_dimension(::AbstractLayer{M, N}) where {M, N} = N


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
function apply!(y::AbstractArray, layer::AbstractLayer, x, ps)
    return layer(y, x, ps)
end


"""
    AbstractExplicitLayer

Abstract supertype for explicit layers.
This type exists mainly for compatibility with Lux.
"""
abstract type AbstractExplicitLayer{N,M} <: AbstractLayer{N,M} end
