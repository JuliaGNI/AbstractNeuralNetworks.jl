using AbstractNeuralNetworks
using JLArrays: JLArray
using Random
using Test

l = Linear(2, 2)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 2 .* i

d = Dense(2, 2, IdentityActivation(); use_bias = false)
p = initialparameters(Random.default_rng(), OneInitializer(), d, CPU(), Float64)

@test l(i, p) == d(i, p)

# `Linear` on a 3-tensor, the same `_mul` as `Dense`: checked against the per-slice loop in both
# element types and on a JLArray.
@testset "Linear on a 3-tensor ($T)" for T in (Float32, Float64)
    l = Linear(4, 3)
    p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), T)
    x = rand(T, 4, 5, 2)

    y = l(x, p)
    y_per_slice = cat((l(x[:, :, k], p) for k in axes(x, 3))...; dims = 3)
    @test y ≈ y_per_slice
    @test eltype(y) == T

    xg = JLArray(x)
    pg = (W = JLArray(p.W),)
    @test Array(l(xg, pg)) ≈ y
end
