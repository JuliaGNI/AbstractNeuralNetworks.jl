using AbstractNeuralNetworks
using Random
using Test

l1 = (
    W = rand(3,3),
    b = rand(3)
)

l2 = (
    W = rand(3,3),
    b = rand(3)
)

ch = (L1 = l1, L2 = l2)

p = NeuralNetworkParameters(ch)

@test hasproperty(p, :L1) == true
@test hasproperty(p, :M1) == false

@test getproperty(p, :L1) == l1
@test_throws ErrorException getproperty(p, :M1)

@test p == NeuralNetworkParameters{keys(ch)}(values(ch))

@test isequal(p, NeuralNetworkParameters(ch))

@test p.L1 == p[:L1] == l1
@test p.L2 == p[:L2] == l2

@test values(p) == (l1,l2)
@test params(p) == ch
