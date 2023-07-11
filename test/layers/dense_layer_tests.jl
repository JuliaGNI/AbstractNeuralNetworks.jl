using AbstractNetworks
using KernelAbstractions
using Random
using Test


_ones!(_, x) = x .= 1

l = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), Float64, l; init=_ones!)

i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(i, p) == l(o1, i, p) == AbstractNetworks.apply!(o2, l, i, p) == 3 .* i


p1 = initialparameters(Random.default_rng(), i, l; init=_ones!)
p2 = initialparameters(Random.default_rng(), Float64, l; init=_ones!)
p3 = initialparameters(Random.default_rng(), CPU(), Float64, l; init=_ones!)
p4 = initialparameters(i, l; init=_ones!)
p5 = initialparameters(Float64, l; init=_ones!)
p6 = initialparameters(CPU(), Float64, l; init=_ones!)

@test p1 == p2 == p3 == p4 == p5 == p6
