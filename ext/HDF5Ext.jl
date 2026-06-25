module HDF5Ext

using HDF5
using AbstractNeuralNetworks
import AbstractNeuralNetworks: h5save, h5load, save, load, NeuralNetworkParameters, params

function _create_group(h5::HDF5.H5DataStore, name)
    if haskey(h5, name)
        g = h5[name]
    else
        g = HDF5.create_group(h5, name)
    end
    return g
end

function h5save(h5::HDF5.H5DataStore, x::AbstractArray, path::AbstractString)
    h5[path] = x
end

function h5save(h5::HDF5.H5DataStore, nt::NamedTuple, path::AbstractString)
    h5group = _create_group(h5, path)
    for (k, v) in pairs(nt)
        h5save(h5group, v, string(k))
    end
end

function save(h5::HDF5.H5DataStore, p::NeuralNetworkParameters)
    h5save(h5, params(p), "/")
end

h5load(h5::HDF5.Dataset) = read(h5)

function h5load(h5group::HDF5.Group)
    paramkeys = Tuple(Symbol.(keys(h5group)))
    paramvals = Tuple(h5load(h5group[k]) for k in keys(h5group))
    NamedTuple{paramkeys}(paramvals)
end

function load(::Type{NeuralNetworkParameters}, h5::HDF5.H5DataStore)
    NeuralNetworkParameters(h5load(h5["/"]))
end

end
