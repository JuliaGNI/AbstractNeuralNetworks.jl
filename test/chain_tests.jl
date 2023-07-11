using AbstractNetworks
using KernelAbstractions
using Random
using Test


i = ones(2)
o = zero(i)

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

p1 = initialparameters(Random.default_rng(), Float64, c; init = OneInitializer())
p2 = initialparameters(Random.default_rng(), CPU(), Float64, c; init = OneInitializer())
p3 = initialparameters(Float64, c; init = OneInitializer())
p4 = initialparameters(CPU(), Float64, c; init = OneInitializer())

@test p1 == p2 == p3 == p4


c = Chain(Dense(2, 2, x -> x))

p = initialparameters(Random.default_rng(), Float64, c; init = OneInitializer())

@test c(i, p) == 3 .* i


c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

p = initialparameters(Random.default_rng(), Float64, c; init = OneInitializer())

@test c(i, p) == 7 .* i


c = Chain(Linear(2, 2),
          Linear(2, 2),
          Linear(2, 2))

p = initialparameters(Random.default_rng(), Float64, c; init = OneInitializer())

@test c(i, p) == 15 .* i
