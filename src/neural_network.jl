abstract type AbstractNeuralNetwork{AT} end


struct NeuralNetwork{AT,MT,PT} <: AbstractNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
end

architecture(nn::NeuralNetwork) = nn.architecture
model(nn::NeuralNetwork) = nn.model
params(nn::NeuralNetwork) = nn.params

function NeuralNetwork(arch::Architecture, model::Model, backend::Backend, ::Type{T}; kwargs...) where {T <: Number}
    # initialize params
    params = initialparameters(backend, T, model; kwargs...)

    # create neural network
    NeuralNetwork(arch, model, params)
end

function NeuralNetwork(arch::Architecture, backend::Backend, ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, Chain(arch), backend, T; kwargs...)
end

function NeuralNetwork(model::Model, backend::Backend, ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(UnknownArchitecture(), model, backend, T; kwargs...)
end

function NeuralNetwork(nn::Union{Architecture, Chain, GridCell}, ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(nn, CPU(), T; kwargs...)
end

function NeuralNetwork(arch::Architecture, model::Model, ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(arch, model, CPU(), T; kwargs...)
end

function NeuralNetwork(backend::Backend, ::Type{T}, model::Union{Architecture, Model}; kwargs...) where {T <: Number}
    NeuralNetwork(model, backend, T; kwargs...)
end

function NeuralNetwork(::Type{T}, backend::Backend, model::Union{Architecture, Model}; kwargs...) where {T <: Number}
    NeuralNetwork(backend, T, model; kwargs...)
end

function NeuralNetwork(::Type{T}, model::Union{Architecture, Model}, backend::Backend; kwargs...) where {T <: Number}
    NeuralNetwork(backend, T, model; kwargs...)
end

function NeuralNetwork(::Type{T}, model::Union{Architecture, Model}; kwargs...) where {T <: Number}
    NeuralNetwork(CPU(), T, model; kwargs...)
end

function NeuralNetwork(backend::Backend, model::Union{Architecture, Model}; kwargs...) 
    NeuralNetwork(backend, Float32, model)
end

function NeuralNetwork(backend::CPU, model::Union{Architecture, Model}; kwargs...) 
    NeuralNetwork(backend, Float64, model)
end

function NeuralNetwork(model::Union{Architecture, Model}, backend::Backend; kwargs...)
    NeuralNetwork(backend, model; kwargs...)
end

function NeuralNetwork(model::Union{Architecture, Model}; kwargs...)
    NeuralNetwork(CPU(), model; kwargs...)
end

function NeuralNetwork(model::Union{Architecture, Model}, ::Type{T}, backend::Backend; kwargs...) where {T <: Number}
    NeuralNetwork(backend, T, model; kwargs...)
end

function NeuralNetwork(backend::Backend, model::Union{Architecture, Model}, ::Type{T}; kwargs...) where {T <: Number}
    NeuralNetwork(backend, T, model; kwargs...)
end

(nn::NeuralNetwork)(x, params) = nn.model(x, params)
(nn::NeuralNetwork)(x) = nn(x, nn.params)

(nn::NeuralNetwork{AT, MT} where {AT, MT<:GridCell})(x, st, params) = nn.model(x, st, params)
(nn::NeuralNetwork{AT, MT} where {AT, MT<:GridCell})(x, params) = nn(x, nn.model.init_st, params)
(nn::NeuralNetwork{AT, MT} where {AT, MT<:GridCell})(x) = nn(x, nn.model.init_st, nn.params)

apply(nn::NeuralNetwork, x, args...) = nn(x, args...)

parameterlength(nn::NeuralNetwork) = parameterlength(nn.model)