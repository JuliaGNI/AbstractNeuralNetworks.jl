"""
    Chain

A chain is a sequence of layers.

A `Chain` can be initialized by passing an arbitrary number of layers
```
Chain(layers...)
```
or a neural network architecture together with a backend and a parameter type:
```
Chain(::Architecture, ::NeuralNetworkBackend, ::Type; kwargs...)
Chain(::Architecture, ::Type; kwargs...)
```
If the backend is omitted, the default backend `CPU()` is chosen.
The keyword arguments will be passed to the `initialparameters` method of each layer.
"""
struct Chain{LT <: Tuple} <: Model
    layers::LT

    function Chain(layers...)
        _layers = Tuple(layers)
        new{typeof(_layers)}(_layers)
    end
end

(model::Chain)(x, ps) = applychain(layers(model), x, ps)

@inline layers(c::Chain) = c.layers
@inline layer(c::Chain, i) = c.layers[i]

Base.length(c::Chain) = length(c.layers)
Base.iterate(c::Chain, i=1) = i > length(c) ? nothing : (layer(c, i), i+1)
Base.eachindex(c::Chain) = 1:length(c)

Base.isequal(c1::Chain, c2::Chain) = isequal(layers(c1), layers(c2))
Base.:(==)(c1::Chain, c2::Chain) = (layers(c1) == layers(c2))

@generated function applychain(layers::Tuple, x::Union{AbstractArray, NamedTuple{(:q, :p), Tuple{AT, AT}}}, ps::Tuple) where {AT <: AbstractArray}
    N = length(fieldtypes((layers)))
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    calls = [:(($(x_symbols[i + 1])) = layers[$i]($(x_symbols[i]), ps[$i])) for i in 1:N]
    push!(calls, :(return $(x_symbols[N + 1])))
    return Expr(:block, calls...)
end

@inline applychain(layers::Tuple, x, ps::Union{NamedTuple,NeuralNetworkParameters}) = applychain(layers, x, values(ps))

function initialparameters(rng::AbstractRNG, initializer::Initializer, model::Chain, backend::NeuralNetworkBackend, ::Type{T}; kwargs...) where T
    keys = Tuple(Symbol("L$(i)") for i in eachindex(model))
    vals = Tuple(initialparameters(rng, initializer, layer, backend, T; kwargs...) for layer in model)
    NeuralNetworkParameters{keys}(vals)
end

function update!(chain::Chain, params::Tuple, grad::Tuple, η::AbstractFloat)
    for (layer, θ, dθ) in zip(chain, params, grad)
        update!(layer, θ, dθ, η)
    end
end

function parameterlength(chain::Chain)
    number_parameters = 0
    for layer in chain.layers
        number_parameters += parameterlength(layer)
    end
    number_parameters
end

Chain(model::Chain, d::AbstractExplicitLayer) = Chain(model.layers..., d)
