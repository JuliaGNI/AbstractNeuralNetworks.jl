using AbstractNeuralNetworks
using Random
using Test


l = Linear(2, 2)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 2 .* i


d = Dense(2, 2, IdentityActivation(); use_bias = false)
p = initialparameters(Random.default_rng(), OneInitializer(), d, CPU(), Float64)

@test l(i, p) == d(i, p)
