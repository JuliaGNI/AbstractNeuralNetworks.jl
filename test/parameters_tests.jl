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

@test p == NeuralNetworkParameters{keys(ch)}(values(ch))

@test p.L1 == l1
@test p.L2 == l2

@test values(p) == (l1,l2)
