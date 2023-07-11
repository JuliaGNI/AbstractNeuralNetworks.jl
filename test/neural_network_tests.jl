using AbstractNetworks
using KernelAbstractions
using Random
using Test


i = ones(2)
o = zero(i)

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

nn1 = NeuralNetwork(c, Float64)
nn1 = NeuralNetwork(c, CPU(), Float64)

@test nn1 == nn2
