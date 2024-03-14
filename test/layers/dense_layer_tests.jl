using AbstractNeuralNetworks
using Random
using Test


i = ones(2)
o1 = zero(i)
o2 = zero(i)


l = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), l, Float64; init = OneInitializer())

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 3 .* i
@test AbstractNeuralNetworks.usebias(l) == true

AbstractNeuralNetworks.update!(l, p, p, 1.0)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 6 .* i


l = Dense(2, 2, x -> x; use_bias = false)
p = initialparameters(Random.default_rng(), l, Float64; init = OneInitializer())

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 2 .* i
@test AbstractNeuralNetworks.usebias(l) == false


p1 = initialparameters(l, Float64; init = OneInitializer())
p2 = initialparameters(l, CPU(), Float64; init = OneInitializer())
p3 = initialparameters(l, Float64; init = OneInitializer())
p4 = initialparameters(l, CPU(), Float64; init = OneInitializer())
p5 = initialparameters(l, Float64; init = OneInitializer(), rng = Random.default_rng())
p6 = initialparameters(l, CPU(), Float64; init = OneInitializer(), rng = Random.default_rng())

@test p1 == p2 == p3 == p4 == p5 == p6
