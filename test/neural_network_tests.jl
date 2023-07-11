using AbstractNetworks
using KernelAbstractions
using Random
using Test


i = ones(2)
o = zero(i)

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

@test_nowarn NeuralNetwork(c, Float64; init = OneInitializer())
@test_nowarn NeuralNetwork(c, CPU(), Float64; init = OneInitializer())
