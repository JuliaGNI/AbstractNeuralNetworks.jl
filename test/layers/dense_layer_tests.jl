using AbstractNetworks
using Random
using Test


i = ones(2)
o1 = zero(i)
o2 = zero(i)


l = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), Float64, l; init = OneInitializer())

@test l(i, p) == l(o1, i, p) == AbstractNetworks.apply!(o2, l, i, p) == 3 .* i
@test usebias(l) == true


l = Dense(2, 2, x -> x; use_bias = false)
p = initialparameters(Random.default_rng(), Float64, l; init = OneInitializer())

@test l(i, p) == l(o1, i, p) == AbstractNetworks.apply!(o2, l, i, p) == 2 .* i
@test usebias(l) == false


p1 = initialparameters(Float64, l; init = OneInitializer())
p2 = initialparameters(CPU(), Float64, l; init = OneInitializer())
p3 = initialparameters(Float64, l; init = OneInitializer())
p4 = initialparameters(CPU(), Float64, l; init = OneInitializer())
p5 = initialparameters(Float64, l; init = OneInitializer(), rng = Random.default_rng())
p6 = initialparameters(CPU(), Float64, l; init = OneInitializer(), rng = Random.default_rng())

@test p1 == p2 == p3 == p4 == p5 == p6
