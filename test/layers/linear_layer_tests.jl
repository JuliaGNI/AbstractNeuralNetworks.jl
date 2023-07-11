using AbstractNetworks
using KernelAbstractions
using Random
using Test


_ones!(_, x) = x .= 1

l = Linear(2, 2)
p = initialparameters(Random.default_rng(), Float64, l; init=_ones!)

i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(i, p) == l(o1, i, p) == AbstractNetworks.apply!(o2, l, i, p) == 3 .* i


d = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), Float64, d)

@test l(i, p) == d(i, p)
