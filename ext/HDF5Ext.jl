module HDF5Ext

using HDF5: HDF5
using AbstractNeuralNetworks: Architecture, CPU, Chain, NeuralNetwork
using NeuralNetworkParameters: NetworkParameters, params
import NeuralNetworkParameters: save, load

# ---------------------------------------------------------------------------
# save / load — the entry points that dispatch on this package's `NeuralNetwork`.
#
# The traversal itself is not here. `NeuralNetworkParameters` walks the parameter set and records
# each group's key order; `GeometricOptimizers` says through `freeparameters`/`rebuild` where each
# structured matrix keeps its numbers, and registers the types so a file loads with no prototype.
# ---------------------------------------------------------------------------

"""
    save(h5::HDF5.H5DataStore, nn::NeuralNetwork)

Save the parameters of `nn` into an already-open HDF5 store.

Extends `save` with a dispatch on `NeuralNetwork`. The parameters themselves are written by
`NeuralNetworkParameters`, which tags each structured leaf with the type to rebuild it as and
records the key order of every group. `save(filename, nn)` is `NeuralNetworkParameters`' own
method: it opens `filename` for writing, calls this method, and returns `filename`.
"""
save(h5::HDF5.H5DataStore, nn::NeuralNetwork) = save(h5, params(nn))

"""
    load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture)
    load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture, prototype)

Load network parameters from an already-open HDF5 store and return a `NeuralNetwork` for `arch`.
`load(NeuralNetwork, filename, arch[, prototype])` is `NeuralNetworkParameters`' own method: it
opens `filename` for reading and calls this method.

The element type is whatever the file holds, so a `Float32` network reloads as `Float32`. The
network is on the CPU; `changebackend(backend, nn)` moves it to a device.

Structured parameters — `StiefelManifold`, `SymmetricMatrix` and the rest — are rebuilt from the
type each was stored under, which `GeometricOptimizers` registers with
`NeuralNetworkParameters.register_parameter_type!`. Pass `prototype`, a parameter set of the right
shape, to rebuild against it instead and skip the registry altogether.
"""
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture, prototype...)
    NeuralNetwork(arch, Chain(arch), load(NetworkParameters, h5, prototype...), CPU())
end

end
