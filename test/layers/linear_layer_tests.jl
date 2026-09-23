using AbstractNeuralNetworks
using JLArrays: JLArray
using Random
using Test
using Zygote: gradient

l = Linear(2, 2)
p = initialparameters(Random.default_rng(), OneInitializer(), l, CPU(), Float64)

i = ones(2)
o1 = zero(i)
o2 = zero(i)

@test l(i, p) == l(o1, i, p) == AbstractNeuralNetworks.apply!(o2, l, i, p) == 2 .* i

d = Dense(2, 2, IdentityActivation(); use_bias = false)
p = initialparameters(Random.default_rng(), OneInitializer(), d, CPU(), Float64)

@test l(i, p) == d(i, p)

# `Linear` on a tensor of rank 3 and 4, the same `_mul` as `Dense`: checked against the layer
# applied to each column `x[:, I]` in both element types, in the gradient, and on a JLArray.
@testset "Linear on rank $(length(dims)) ($T)" for T in (Float32, Float64),
    dims in ((4, 5, 2), (4, 5, 2, 3))

    l = Linear(4, 3)
    p = initialparameters(Random.default_rng(), GlorotUniform(), l, CPU(), T)
    x = rand(T, dims...)
    columns = CartesianIndices(Base.tail(size(x)))

    y = l(x, p)
    @test y ≈ stack(l(x[:, I], p) for I in columns)
    @test eltype(y) == T

    g = gradient((x, p) -> sum(l(x, p)), x, p)
    g_per_column = gradient((x, p) -> sum(sum(l(x[:, I], p)) for I in columns), x, p)
    @test g[1] ≈ g_per_column[1]
    @test all(g[2][k] ≈ g_per_column[2][k] for k in keys(p))

    xg = JLArray(x)
    pg = (W = JLArray(p.W),)
    @test Array(l(xg, pg)) ≈ y
end
