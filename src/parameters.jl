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


function _name(h5::H5DataStore)
    name = HDF5.name(h5)
    name = name[findlast(isequal('/'), name)+1:end]
end


function _create_group(h5::H5DataStore, name)
    if haskey(h5, name)
        g = h5[name]
    else
        g = create_group(h5, name)
    end
    return g
end

function h5save(h5::H5DataStore, x::AbstractArray, path::AbstractString)
    h5[path] = x
end

function h5save(h5::H5DataStore, nt::NamedTuple, path::AbstractString)
    h5group = _create_group(h5, path)
    for (k,v) in pairs(nt)
        h5save(h5group, v, string(k))
    end
end

function save(h5::H5DataStore, p::NeuralNetworkParameters)
    h5save(h5, params(p), "/")
end


h5load(h5::HDF5.Dataset) = read(h5)

function h5load(h5group::HDF5.Group)
    paramkeys = Tuple(Symbol.(keys(h5group)))
    paramvals = Tuple(h5load(h5group[k]) for k in keys(h5group))

    NamedTuple{paramkeys}(paramvals)
end

function load(::Type{NeuralNetworkParameters}, h5::H5DataStore)
    NeuralNetworkParameters(h5load(h5["/"]))
end
