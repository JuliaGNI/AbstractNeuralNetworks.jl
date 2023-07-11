module AbstractNetworks

    using Base: Callable
    using KernelAbstractions
    using LinearAlgebra
    using Random


    include("add.jl")
    include("zero_vector.jl")

    include("activation.jl")

    include("architecture.jl")


    export OneInitializer, ZeroInitializer

    include("initializer.jl")


    export initialparameters

    include("model.jl")


    export Dense, Linear
    export usebias

    include("layers/abstract.jl")
    include("layers/dense.jl")
    include("layers/linear.jl")


    export Chain

    include("chain.jl")

end
