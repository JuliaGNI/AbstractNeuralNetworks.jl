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
    include("utils/array_named_tuple.jl")


    export Activation, GenericActivation, IdentityActivation, SigmoidActivation

    include("activation.jl")

    include("architecture.jl")


    # The parameter container lives in `NeuralNetworkParameters` now, as `NetworkParameters`, along
    # with the tree walks and the HDF5 path that used to be duplicated here. This package is a
    # consumer of it; 0.7 removed the `NeuralNetworkParameters` name from here entirely rather than
    # leaving an alias behind, so that one type has one name across the ecosystem.
    #
    # `import` rather than `using ... :` for the five names that are extended or reached through this
    # module: `params` gains a `NeuralNetwork` method below, and downstream packages add methods to
    # the four storage generics via `import AbstractNeuralNetworks: h5save, save, load`.
    using NeuralNetworkParameters: NetworkParameters, ParameterSet, mapparameters
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
