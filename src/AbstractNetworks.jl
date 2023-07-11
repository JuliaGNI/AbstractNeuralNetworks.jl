module AbstractNetworks

    using Base: Callable
    using KernelAbstractions
    using LinearAlgebra
    using Random


    include("abstract_initializer.jl")
    include("add.jl")
    include("zero_vector.jl")

    include("activation.jl")


    export initialparameters

    include("model.jl")


    export Dense, Linear

    include("layers/abstract.jl")
    include("layers/dense.jl")
    include("layers/linear.jl")


    export Chain

    include("chain.jl")

end
