module AbstractNeuralNetworks

    using KernelAbstractions
    using LinearAlgebra
    using Random

    export CPU, GPU
    
    include("utils/add.jl")
    include("utils/zero_vector.jl")

    include("activation.jl")

    include("architecture.jl")


    export OneInitializer, ZeroInitializer, GlorotUniform

    include("initializer.jl")


    export initialparameters
    export parameterlength

    include("model.jl")


    export Dense, Linear

    include("layers/abstract.jl")
    include("layers/dense.jl")
    include("layers/linear.jl")

    export Chain

    include("chain.jl")

    export Recurrent, LSTM, IdentityCell, GRU

    include("cells/abstract.jl")
    include("cells/recurrent.jl")
    include("cells/lstm.jl")
    include("cells/identity.jl")
    include("cells/gru.jl")

    export GridCell

    include("grid.jl")

    export AbstractNeuralNetwork
    export NeuralNetwork

    include("neural_network.jl")

end
