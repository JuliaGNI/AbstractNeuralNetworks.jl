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


    export NeuralNetwork

    include("neural_network.jl")

end
