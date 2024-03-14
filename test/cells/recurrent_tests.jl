using AbstractNeuralNetworks
using Test
using Random

import AbstractNeuralNetworks: usebias

c = Recurrent(2, 3, 1, 2, tanh)

@test size(c) == (2, 3, 1, 2)
@test usebias(c) == true

p = initialparameters(Random.default_rng(), c, Float64)
x = [4,5]
st = [1,2,3]

y, nst = c(x, st, p)

@test length(y) == 1
@test length(nst) == 2


# Tests for all combinations

c2 = Recurrent(2, 3, 0, 2, tanh)
p = initialparameters(Random.default_rng(), c2, Float64)
y, nst = c2(x, st, p)

c3 = Recurrent(2, 3, 1, 2, tanh; use_bias = false)
p = initialparameters(Random.default_rng(), c3, Float64)
y, nst = c3(x, st, p)

c4 = Recurrent(2, 3, 0, 2, tanh; use_bias = false)
p = initialparameters(Random.default_rng(), c4, Float64)
y, nst = c4(x, st, p)