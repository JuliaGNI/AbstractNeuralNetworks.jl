module AbstractNeuralNetworks

using KernelAbstractions
using GPUArraysCore: AbstractGPUArray
using LinearAlgebra
using StaticArrays
using Random
using ZygoteRules

export CPU, GPU

include("utils/add.jl")
include("utils/zero_vector.jl")
include("utils/named_tuple_of_arrays.jl")

export Activation, GenericActivation, IdentityActivation, SigmoidActivation

include("activation.jl")

include("architecture.jl")

# The parameter container `NetworkParameters`, its tree walks and its HDF5 storage (`h5save`,
# `h5load`, `save`, `load`) come from `NeuralNetworkParameters`. This package adds a
# `NeuralNetwork` method to `params` below, and `ext/HDF5Ext.jl` adds `NeuralNetwork` methods to
# `save` and `load`. All five names are reachable as `AbstractNeuralNetworks.<name>`.
using NeuralNetworkParameters: NetworkParameters, mapparameters, storage_gradient,
                               map_cotangent
import NeuralNetworkParameters: params, h5save, h5load, save, load

export params

include("static_cpu_backend.jl")

export NeuralNetworkBackend, networkbackend

include("neural_network_backend.jl")

export OneInitializer, ZeroInitializer, GlorotUniform

include("initializer.jl")

export initialparameters
export parameterlength

include("model.jl")

export Dense, Linear, Affine
export input_dimension, output_dimension

include("layers/abstract.jl")
include("layers/dense.jl")
include("layers/affine.jl")
include("layers/linear.jl")

export Chain

include("chain.jl")

include("pullback_for_applychain.jl")

export AbstractNeuralNetwork
export NeuralNetwork

include("neural_network.jl")

include("losses.jl")

export NetworkLoss, FeedForwardLoss

include("pullback.jl")

export AbstractPullback

export changebackend
include("utils/changebackend.jl")
end
