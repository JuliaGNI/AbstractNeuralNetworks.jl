using AbstractNeuralNetworks
using Random
using Test

i = ones(2)
o1 = zero(i)
o2 = zero(i)

l = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

@test l(i, p) ==  3 .* i
@test AbstractNeuralNetworks.usebias(l) == true

AbstractNeuralNetworks.update!(l, p, p, 1.0)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 6 .* i

l = Dense(2, 2, x -> x; use_bias = false)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 2 .* i
@test AbstractNeuralNetworks.usebias(l) == false