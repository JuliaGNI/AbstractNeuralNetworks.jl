using AbstractNeuralNetworks
using Random
using Test

# NeuralNetwork with Chain

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

@test_nowarn NeuralNetwork(c, Float64; init = OneInitializer())
@test_nowarn NeuralNetwork(c, CPU(), Float64; init = OneInitializer())

nn = NeuralNetwork(c, Float64; init = OneInitializer())

@test params(nn) == nn.params
@test model(nn) == c

x = [1,2]

@test_nowarn nn(x)
@test_nowarn nn(x, nn.params)

# NeuralNetwork with  GridCell

g = GridCell([Recurrent(2, 2, 2, 2, tanh) Recurrent(2, 2, 2, 2, tanh);
              Recurrent(2, 2, 2, 2, tanh) Recurrent(2, 2, 2, 2, tanh)])

@test_nowarn NeuralNetwork(g, Float64; init = OneInitializer())
@test_nowarn NeuralNetwork(g, CPU(), Float64; init = OneInitializer())

nn = NeuralNetwork(g, Float64; init = OneInitializer())

x = [[1,2], [3,4]]
st = [[1,2], [3,4]]

@test_nowarn nn(x)
@test_nowarn nn(x, nn.params)
@test_nowarn nn(x, st, nn.params)