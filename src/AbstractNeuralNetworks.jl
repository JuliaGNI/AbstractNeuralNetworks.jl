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


    export Activation, GenericActivation, IdentityActivation, SigmoidActivation

    include("activation.jl")

    include("architecture.jl")


    # The parameter container lives in `NeuralNetworkParameters` now, along with the tree walks and
    # the HDF5 path that used to be duplicated here. The import is selective on purpose: a bare
    # `using NeuralNetworkParameters` would bind the *module* name, and that is exactly the name the
    # compatibility alias in `parameters.jl` needs.
    #
    # `import` rather than `using ... :` for the five names that are extended or reached through this
    # module: `params` gains a `NeuralNetwork` method below, and downstream packages add methods to
    # the four storage generics via `import AbstractNeuralNetworks: h5save, save, load`.
    import NeuralNetworkParameters: params, h5save, h5load, save, load
    using NeuralNetworkParameters: NetworkParameters, mapparameters

    export params

    include("parameters.jl")

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

    export Recurrent, LSTM, IdentityCell, GRU, GridCell

    include("cells/abstract.jl")
    include("cells/recurrent.jl")
    include("cells/lstm.jl")
    include("cells/identity.jl")
    include("cells/gru.jl")
    include("cells/grid.jl")

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
