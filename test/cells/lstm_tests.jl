using AbstractNeuralNetworks
using Test
using Random

c = LSTM(2, 3)

@test size(c) == (2, 6, 3, 6)

p = initialparameters(Random.default_rng(), Float64, c)

x = [4,5]
st = [1,2,3, 4, 5, 6]

y, nst = c(x, st, p)
@test length(y) == 3
@test length(st) == 6
