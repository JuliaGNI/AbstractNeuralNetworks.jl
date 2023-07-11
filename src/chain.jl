

struct Chain{LT <: Tuple} <: Model
    layers::LT

    function Chain(layers...)
        _layers = Tuple(layers)
        new{typeof(_layers)}(_layers)
    end
end

(model::Chain)(x, ps) = applychain(model.layers, x, ps)

@inline layers(c::Chain) = c.layers
@inline layer(c::Chain, i) = c.layers[i]

Base.length(c::Chain) = length(c.layers)
Base.iterate(c::Chain, i=1) = i > length(c) ? nothing : (layer(c, i), i+1)


@generated function applychain(layers::Tuple, x::AbstractArray, ps::Tuple)
    N = length(fieldtypes((layers)))
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    calls = [:(($(x_symbols[i + 1])) = layers[$i]($(x_symbols[i]), ps[$i])) for i in 1:N]
    push!(calls, :(return $(x_symbols[N + 1])))
    return Expr(:block, calls...)
end


function initialparameters(rng::AbstractRNG, backend::Backend, ::Type{T}, model::Chain; kwargs...) where {T}
    Tuple(initialparameters(rng, backend, T, layer; kwargs...) for layer in model)
end

function initialparameters(rng::AbstractRNG, x::AbstractArray, model::Chain; kwargs...)
    Tuple(initialparameters(rng, x, layer; kwargs...) for layer in model)
end
