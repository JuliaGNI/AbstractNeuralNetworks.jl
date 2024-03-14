using AbstractNeuralNetworks
using Random
using Test


i = ones(2)
o = zero(i)

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

@test eachindex(c) == 1:3

p1 = initialparameters(Random.default_rng(), c, Float64; init = OneInitializer())
p2 = initialparameters(Random.default_rng(), c, CPU(), Float64; init = OneInitializer())
p3 = initialparameters(c, Float64; init = OneInitializer())
p4 = initialparameters(c, CPU(), Float64; init = OneInitializer())
p5 = initialparameters(c, Float64; init = OneInitializer(), rng = Random.default_rng())
p6 = initialparameters(c, CPU(), Float64; init = OneInitializer(), rng = Random.default_rng())

@test p1 == p2 == p3 == p4 == p5 == p6


c = Chain(Dense(2, 2, x -> x))

p = initialparameters(c, Float64; init = OneInitializer())

@test c(i, p) == 3 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 6 .* i


c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

p = initialparameters(c, Float64; init = OneInitializer())

@test c(i, p) == 7 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 26 .* i


c = Chain(Linear(2, 2),
          Linear(2, 2),
          Linear(2, 2))

p = initialparameters(c, Float64; init = OneInitializer())

@test c(i, p) == 15 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 106 .* i
