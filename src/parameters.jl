"""
    NeuralNetworkParameters

This struct stores the parameters of a neural network.
In essence, it is just a wrapper around a `NamedTuple` of `NamedTuple`
that provides some context, e.g., for storing parameters to file.
"""
struct NeuralNetworkParameters{Keys, ValueTypes}
    params::NamedTuple{Keys, ValueTypes}
end

NeuralNetworkParameters{Keys}(values) where {Keys} = NeuralNetworkParameters(NamedTuple{Keys}(values))

params(p::NeuralNetworkParameters) = p.params


@inline function Base.hasproperty(::NeuralNetworkParameters{Keys}, s::Symbol) where {Keys}
    s ∈ Keys || hasfield(NeuralNetworkParameters, s)
end

@inline function Base.getproperty(p::NeuralNetworkParameters{Keys}, s::Symbol) where {Keys}
    if s ∈ Keys
        return getfield(p, :params)[s]
    else
        return getfield(p, s)
    end
end

Base.getindex(p::NeuralNetworkParameters, args...) = getindex(params(p), args...)
Base.keys(p::NeuralNetworkParameters) = keys(params(p))
Base.values(p::NeuralNetworkParameters) = values(params(p))
Base.isequal(p1::NeuralNetworkParameters, p2::NeuralNetworkParameters) = isequal(params(p1), params(p2))
Base.:(==)(p1::NeuralNetworkParameters, p2::NeuralNetworkParameters) = (params(p1) == params(p2))
