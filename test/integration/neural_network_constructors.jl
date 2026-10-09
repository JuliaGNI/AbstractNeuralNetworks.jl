using AbstractNeuralNetworks
using Test
import Random

using AbstractNeuralNetworks: params

function test_different_cpu_constructors(::Type{T}) where {T <: Number}
    model = Chain(Dense(4, 5, tanh), Linear(5, 4))
    Random.seed!(123)
    nn1 = NeuralNetwork(model, CPU(), T)
    Random.seed!(123)
    nn2 = NeuralNetwork(model, T)
    Random.seed!(123)
    nn3 = T == Float64 ? NeuralNetwork(model, CPU()) : nn2
    Random.seed!(123)
    nn4 = T == Float64 ? NeuralNetwork(model) : nn3

    @test params(nn1) == params(nn2) == params(nn3) == params(nn4)
end

test_different_cpu_constructors(Float16)
test_different_cpu_constructors(Float32)
test_different_cpu_constructors(Float64)

# `NeuralNetwork` stores `PT <: NetworkParameters`. A bare layer's `initialparameters` returns the
# plain `NamedTuple` its functor takes, so it cannot be stored -- every entry point should say that
# rather than raising a `MethodError` against the inner four-argument constructor.

layer = Dense(4, 5, tanh)

@test_throws ArgumentError NeuralNetwork(layer, Float64)
@test_throws ArgumentError NeuralNetwork(layer, CPU(), Float64)
@test_throws ArgumentError NeuralNetwork(layer, CPU())
@test_throws ArgumentError NeuralNetwork(layer)

# `Chain` forwards keyword arguments to each layer's `initialparameters`, as its docstring promises,
# so a layer has to tolerate them.

@test_nowarn NeuralNetwork(Chain(layer), Float64; unused_by_dense = 1)
