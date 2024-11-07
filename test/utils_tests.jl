using AbstractNeuralNetworks: add!
using AbstractNeuralNetworks: ZeroVector
using LinearAlgebra
using Random
using Test


a = rand(3)
b = rand(3)
x = zeros(3)

add!(x, a, b)

@test x == a + b

b .= x

add!(x, a)

@test x == a + b



a = rand(3,4)
b = rand(3,4)
x = zeros(3,4)

add!(x, a, b)

@test x == a + b

b .= x

add!(x, a)

@test x == a + b


x = rand(5)
y = copy(x)
z = ZeroVector(x)

@test z == ZeroVector(Float64, 5)

@test eltype(z) == Float64
@test length(z) == 5
@test size(z) == (5,)
@test axes(z) == (1:5,)

@test z[1] == 0
@test z[5] == 0

@test_throws AssertionError z[0] == 0
@test_throws AssertionError z[6] == 0

@test add!(x,z) == y

mul!(x, rand(5,5), z)

@test x == zero(x)
