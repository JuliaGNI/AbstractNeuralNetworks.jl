using AbstractNeuralNetworks
using Test
using Random

c = GRU(2, 3)

@test size(c) == (2, 3, 3, 3)

p = initialparameters(Random.default_rng(), c, Float64)

x = [4,5]
st = [1,2,3]

y, nst = c(x, st, p)
@test y == nst
@test length(y) == 3

