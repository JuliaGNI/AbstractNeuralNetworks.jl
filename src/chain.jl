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

# `x` is deliberately untyped, matching the `ps::NetworkParameters` method below:
# a chain applies whatever its layers accept, and it is the layers that should say what that is.
# This used to be `Union{AbstractArray, NamedTuple{(:q, :p), Tuple{AT, AT}}}`, which both leaked
# Hamiltonian vocabulary into a generic package (issue #31) and forced downstream packages to commit
# type piracy to push anything else through a `Chain`.
@generated function applychain(layers::Tuple, x, ps::Tuple)
    N = length(fieldtypes((layers)))
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    calls = [:(($(x_symbols[i + 1])) = layers[$i]($(x_symbols[i]), ps[$i])) for i in 1:N]
    push!(calls, :(return $(x_symbols[N + 1])))
    return Expr(:block, calls...)
end

# A whole set of parameters is a `NetworkParameters`, which is what `initialparameters` above returns
# and what every package in this ecosystem passes around. The `Tuple` method is the one that does the
# work — `values(ps)` hands it the layers in order — and it stays untyped in its layers, because a
# chain applies whatever its layers accept.
@inline applychain(layers::Tuple, x, ps::NetworkParameters) = applychain(layers, x, values(ps))

# The bare `NamedTuple` a *reverse pass* produces, which is the reason this method exists and the only
# reason. `NeuralNetworkParameters`' `ZygoteRules.pullback(f, ::NetworkParameters)` seeds the reverse
# pass with the wrapped `NamedTuple` rather than the container, because that is what yields a tangent
# keyed by the layers rather than a tangent for the wrapper's one field — so a chain differentiated
# with respect to its parameters is *called* with the `NamedTuple`. `test/custom_pullback_test.jl` is
# what fails without this.
#
# Two methods and not one on a union of the two types: they answer different questions that happen to
# share a body. This one is not an invitation to pass a bare `NamedTuple` — nothing in this ecosystem
# does outside a reverse pass — and writing it out says which of the two shapes each caller is in.
@inline applychain(layers::Tuple, x, ps::NamedTuple) = applychain(layers, x, values(ps))

function initialparameters(rng::AbstractRNG, initializer::Initializer, model::Chain, backend::NeuralNetworkBackend, ::Type{T}; kwargs...) where T
    keys = Tuple(Symbol("L$(i)") for i in eachindex(model))
    vals = Tuple(initialparameters(rng, initializer, layer, backend, T; kwargs...) for layer in model)
    NetworkParameters(NamedTuple{keys}(vals))
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
