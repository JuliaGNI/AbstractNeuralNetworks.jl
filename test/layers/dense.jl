using AbstractNeuralNetworks
using JLArrays: JLArray
using Random
using Test
using Zygote: gradient

Random.seed!(123)

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

# `Dense` on a tensor of rank 3 and 4, checked against the layer applied to each column `x[:, I]`
# in both element types, in the gradient, and on a JLArray, the device stand-in reachable from
# the test suite.
@testset "Dense on rank $(length(dims)) ($T, use_bias = $use_bias)" for T in (Float32, Float64),
    use_bias in (true, false), dims in ((4, 5, 2), (4, 5, 2, 3))

    d = Dense(4, 3, tanh; use_bias)
    p = initialparameters(Random.default_rng(), GlorotUniform(), d, CPU(), T)
    x = rand(T, dims...)
    columns = CartesianIndices(Base.tail(size(x)))

    y = d(x, p)
    @test y ≈ stack(d(x[:, I], p) for I in columns)
    @test eltype(y) == T

    g = gradient((x, p) -> sum(d(x, p)), x, p)
    g_per_column = gradient((x, p) -> sum(sum(d(x[:, I], p)) for I in columns), x, p)
    @test g[1] ≈ g_per_column[1]
    @test all(g[2][k] ≈ g_per_column[2][k] for k in keys(p))

    xg = JLArray(x)
    pg = use_bias ? (W = JLArray(p.W), b = JLArray(p.b)) : (W = JLArray(p.W),)
    @test Array(d(xg, pg)) ≈ y
end
