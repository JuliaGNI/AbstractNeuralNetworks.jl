using AbstractNeuralNetworks
using AbstractNeuralNetworks: IdentityActivation
using Random
using Test


i = ones(2)
o = zero(i)

c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

@test parameterlength(c) == 18

@test eachindex(c) == 1:3

p2 = initialparameters(Random.default_rng(), OneInitializer(), c, CPU(), Float64)

@test Chain(Dense(2, 2, IdentityActivation()), Dense(2, 2, IdentityActivation())) == Chain(Chain(Dense(2, 2, IdentityActivation())), Dense(2, 2, IdentityActivation()))

c = Chain(Dense(2, 2, x -> x))

@test parameterlength(c) == 6

p = initialparameters(Random.default_rng(), OneInitializer(), c, CPU(), Float64)

@test c(i, p) == 3 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 6 .* i


c = Chain(Dense(2, 2, x -> x),
          Dense(2, 2, x -> x))

p = initialparameters(Random.default_rng(), OneInitializer(), c, CPU(), Float64)

@test c(i, p) == 7 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 26 .* i


c = Chain(Affine(2, 2),
          Affine(2, 2),
          Affine(2, 2))

p = initialparameters(Random.default_rng(), OneInitializer(), c, CPU(), Float64)

@test c(i, p) == 15 .* i

AbstractNeuralNetworks.update!(c, p, p, 1.0)

@test c(i, p) == 106 .* i
