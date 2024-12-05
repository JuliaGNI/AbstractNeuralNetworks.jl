using AbstractNeuralNetworks
using StaticArrays
import Random
Random.seed!(123)

c = Chain(Dense(2, 10, tanh), Dense(10, 1, tanh))
nn = NeuralNetwork(c, AbstractNeuralNetworks.CPUStatic())
input = @SVector rand(2)
@test typeof(nn(input)) <: StaticArray