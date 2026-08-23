using AbstractNeuralNetworks
using Random
using Test

import AbstractNeuralNetworks: params, model

# NeuralNetwork with Chain

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

@test_nowarn NeuralNetwork(c, Float64; initializer = OneInitializer())
@test_nowarn NeuralNetwork(c, CPU(), Float64; initializer = OneInitializer())

nn = NeuralNetwork(c, Float64; initializer = OneInitializer())

@test params(nn) == nn.params
@test model(nn) == c

x = [1,2]

@test_nowarn nn(x)
@test_nowarn nn(x, nn.params)
