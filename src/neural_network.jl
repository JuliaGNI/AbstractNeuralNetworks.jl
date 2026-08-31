abstract type AbstractNeuralNetwork{AT} end

"""
    NeuralNetwork <: AbstractNeuralNetwork

`Neuralnetwork` stores the [`Architecture`](@ref), [`Model`](@ref), neural network paramters and backend of the system.

# Implementation

See [`NeuralNetworkBackend`](@ref) for the backend.
"""
struct NeuralNetwork{AT, MT, PT <: NetworkParameters, BT <: NeuralNetworkBackend} <:
       AbstractNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
    backend::BT
end

architecture(nn::NeuralNetwork) = nn.architecture
model(nn::NeuralNetwork) = nn.model
params(nn::NeuralNetwork) = nn.params
networkbackend(nn::NeuralNetwork) = nn.backend

function NeuralNetwork(arch::Architecture, model::Model, backend::NeuralNetworkBackend,
        ::Type{T}; rng = Random.default_rng(),
        initializer = DefaultInitializer(), kwargs...) where {T <: Number}
    # initialize params
    params = initialparameters(rng, initializer, model, backend, T; kwargs...)

    # `NeuralNetwork` stores `PT <: NetworkParameters`, so a model is only usable here if its
    # `initialparameters` returns one. `Chain` does; a bare layer returns the plain `NamedTuple`
    # it hands to its own functor. Say so, rather than letting the inner constructor raise a
    # `MethodError` that names neither the model nor the reason.
    params isa NetworkParameters || throw(ArgumentError(
        "initialparameters for model type $(typeof(model)) returned a $(typeof(params)), " *
        "but NeuralNetwork requires a NetworkParameters. Wrap the model in a `Chain`, or have " *
        "its initialparameters return a `NetworkParameters`."))

    # create neural network
    NeuralNetwork(arch, model, params, backend)
end

function NeuralNetwork(arch::Architecture, backend::NeuralNetworkBackend,
        ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, Chain(arch), backend, T; kwargs...)
end

function NeuralNetwork(model::Model, backend::NeuralNetworkBackend, ::Type{T}; kwargs...) where {T <:
                                                                                                 Number}
    NeuralNetwork(UnknownArchitecture(), model, backend, T; kwargs...)
end

function NeuralNetwork(nn::Union{Architecture, Model}, ::Type{T}; kwargs...) where {T <:
                                                                                    Number}
    NeuralNetwork(nn, CPU(), T; kwargs...)
end

function NeuralNetwork(arch::Architecture, model::Model, ::Type{T}; kwargs...) where {T <:
                                                                                      Number}
    NeuralNetwork(arch, model, CPU(), T; kwargs...)
end

function NeuralNetwork(model::Union{Architecture, Model}, backend::GPU; kwargs...)
    NeuralNetwork(model, backend, Float32; kwargs...)
end

function NeuralNetwork(model::Union{Architecture, Model}, backend::Union{CPU, CPUStatic}; kwargs...)
    NeuralNetwork(model, backend, Float64; kwargs...)
end

function NeuralNetwork(model::Union{Architecture, Model}, backend::NeuralNetworkBackend; kwargs...)
    error("Default type for $(backend) not defined.")
end

function NeuralNetwork(model::Union{Architecture, Model}; kwargs...)
    NeuralNetwork(model, CPU(); kwargs...)
end

(nn::NeuralNetwork)(x, params) = nn.model(x, params)
(nn::NeuralNetwork)(x) = nn(x, nn.params)

apply(nn::NeuralNetwork, x, args...) = nn(x, args...)

parameterlength(nn::NeuralNetwork) = parameterlength(nn.model)
