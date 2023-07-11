using AbstractNetworks
using KernelAbstractions
using Random
using Test


_ones!(_, x) = x .= 1

i = ones(2)
o1 = zero(i)
o2 = zero(i)


l = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), Float64, l; init=_ones!)

@test l(i, p) == l(o1, i, p) == AbstractNetworks.apply!(o2, l, i, p) == 3 .* i
@test usebias(l) == true


l = Dense(2, 2, x -> x; use_bias = false)
p = initialparameters(Random.default_rng(), Float64, l; init=_ones!)

@test l(i, p) == l(o1, i, p) == AbstractNetworks.apply!(o2, l, i, p) == 2 .* i
@test usebias(l) == false


p1 = initialparameters(Random.default_rng(), Float64, l; init=_ones!)
p2 = initialparameters(Random.default_rng(), CPU(), Float64, l; init=_ones!)
p3 = initialparameters(Float64, l; init=_ones!)
p4 = initialparameters(CPU(), Float64, l; init=_ones!)

@test p1 == p2 == p3 == p4
