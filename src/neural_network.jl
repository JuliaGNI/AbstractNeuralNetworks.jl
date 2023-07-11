
struct NeuralNetwork{AT,MT,PT}
    architecture::AT
    model::MT
    params::PT
end

function NeuralNetwork(arch::Architecture, backend::Backend, ::Type{T}; kwargs...) where {T}
    # create model
    model = Chain(arch)

    # initialize params
    params = initialparameters(chain, backend, T; kwargs...)

    # create neural network
    NeuralNetwork(arch, model, params)
end


(nn::NeuralNetwork)(x, params) = nn.model(x, params)
(nn::NeuralNetwork)(x) = nn(x, nn.params)

apply(nn::NeuralNetwork, x, args...) = nn(x, args...)
