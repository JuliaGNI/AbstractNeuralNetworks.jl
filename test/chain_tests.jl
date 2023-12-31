using AbstractNeuralNetworks
using Random
using Test


i = ones(2)
o = zero(i)

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

@test eachindex(c) == 1:3

p1 = initialparameters(Random.default_rng(), Float64, c; init = OneInitializer())
p2 = initialparameters(Random.default_rng(), CPU(), Float64, c; init = OneInitializer())
p3 = initialparameters(Float64, c; init = OneInitializer())
p4 = initialparameters(CPU(), Float64, c; init = OneInitializer())
p5 = initialparameters(Float64, c; init = OneInitializer(), rng = Random.default_rng())
p6 = initialparameters(CPU(), Float64, c; init = OneInitializer(), rng = Random.default_rng())

@test p1 == p2 == p3 == p4 == p5 == p6


c = Chain(Dense(2, 2, x -> x))

p = initialparameters(Float64, c; init = OneInitializer())

@test c(i, p) == 3 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 6 .* i


c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

p = initialparameters(Float64, c; init = OneInitializer())

@test c(i, p) == 7 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 26 .* i


c = Chain(Linear(2, 2),
          Linear(2, 2),
          Linear(2, 2))

p = initialparameters(Float64, c; init = OneInitializer())

@test c(i, p) == 15 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 106 .* i
