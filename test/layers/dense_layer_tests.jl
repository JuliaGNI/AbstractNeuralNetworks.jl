using AbstractNeuralNetworks
using JLArrays: JLArray
using Random
using Test

i = ones(2)
o1 = zero(i)
o2 = zero(i)

l = Dense(2, 2, x -> x)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

@test l(i, p) == 3 .* i
@test AbstractNeuralNetworks.usebias(l) == true

AbstractNeuralNetworks.update!(l, p, p, 1.0)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 6 .* i

l = Dense(2, 2, x -> x; use_bias = false)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 2 .* i
@test AbstractNeuralNetworks.usebias(l) == false

# `Dense` on a 3-tensor: a reshape and one GEMM against every slice, no kernel. Checked against the
# per-slice loop in both element types and on a JLArray, the device stand-in reachable from the
# test suite.
@testset "Dense on a 3-tensor ($T, use_bias = $use_bias)" for T in (Float32, Float64),
    use_bias in (true, false)

    d = Dense(4, 3, tanh; use_bias)
    p = initialparameters(Random.default_rng(), OneInitializer(), d, CPU(), T)
    x = rand(T, 4, 5, 2)

    y = d(x, p)
    y_per_slice = cat((d(x[:, :, k], p) for k in axes(x, 3))...; dims = 3)
    @test y ≈ y_per_slice
    @test eltype(y) == T

    xg = JLArray(x)
    pg = use_bias ? (W = JLArray(p.W), b = JLArray(p.b)) : (W = JLArray(p.W),)
    @test Array(d(xg, pg)) ≈ y
end
