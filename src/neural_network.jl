abstract type AbstractNeuralNetwork{AT} end


struct NeuralNetwork{AT,MT,PT} <: AbstractNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
end

architecture(nn::NeuralNetwork) = nn.architecture
model(nn::NeuralNetwork) = nn.model
params(nn::NeuralNetwork) = nn.params

function NeuralNetwork(arch::Architecture, model::Chain, backend::Backend, ::Type{T}; kwargs...) where {T}
    # initialize params
    params = initialparameters(backend, T, model; kwargs...)

    # create neural network
    NeuralNetwork(arch, model, params)
end

function NeuralNetwork(arch::Architecture, backend::Backend, ::Type{T}; kwargs...) where {T}
    NeuralNetwork(arch, Chain(arch), backend, T; kwargs...)
end

function NeuralNetwork(model::Chain, backend::Backend, ::Type{T}; kwargs...) where {T}
    NeuralNetwork(UnknownArchitecture(), model, backend, T; kwargs...)
end

function NeuralNetwork(nn::Union{Architecture,Chain}, ::Type{T}; kwargs...) where {T}
    NeuralNetwork(nn, CPU(), T; kwargs...)
end


(nn::NeuralNetwork)(x, params) = nn.model(x, params)
(nn::NeuralNetwork)(x) = nn(x, nn.params)

apply(nn::NeuralNetwork, x, args...) = nn(x, args...)
