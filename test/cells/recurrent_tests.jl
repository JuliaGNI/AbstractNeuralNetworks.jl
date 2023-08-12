using AbstractNeuralNetworks
using Test
using Random

c = Recurrent(2, 3, 1, 2, tanh)

@test size(c) == (2, 3, 1, 2)

p = initialparameters(Random.default_rng(), Float64, c)
x = [4,5]
st = [1,2,3]

y, nst = c(x, st, p)

@test length(y) == 1
@test length(nst) == 2