using AbstractNeuralNetworks
using Random
using Test


l = Linear(2, 2)
p = initialparameters(Float64, l; init = OneInitializer(), rng = Random.default_rng())

i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 3 .* i


d = Dense(2, 2, x -> x)
p = initialparameters(Float64, d)

@test l(i, p) == d(i, p)
