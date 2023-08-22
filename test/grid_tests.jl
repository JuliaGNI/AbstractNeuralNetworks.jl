using AbstractNeuralNetworks
using Random
using Test

rng = Random.seed!(1234)

input = [[2,2],[2]]
st = [[3,4], [7,8,9], [1, 2, 3, 4, 5, 6, 7, 8]]

c1 = Recurrent(2, 2, 2, 2, tanh)
c2 = Recurrent(1, 2, 1, 1, tanh)
c3 = Recurrent(2, 3, 2, 2, tanh)
c4 = IdentityCell()
c5 = GRU(2, 8)
c6 = LSTM(1, 4)

p1 = initialparameters(rng, Float64, c1)
p3 = initialparameters(rng, Float64, c3)
p5 = initialparameters(rng, Float64, c5)
p2 = initialparameters(rng, Float64, c2)
p4 = initialparameters(rng, Float64, c4)
p6 = initialparameters(rng, Float64, c6)

params = [p1 p2 ; p3 p4 ; p5 p6]

g = GridCell( [c1  c2;
               c3  c4;
               c5  c6])

@test Tuple([e for e in eachindex(g)]) == ((1,1),(2,1), (3,1), (1,2), (2,2), (3,2))

rng = Random.seed!(1234)
@test params == initialparameters(rng, Float64, g)

@test AbstractNeuralNetworks.cell(g, 1, 1) == c1
@test AbstractNeuralNetworks.cell(g, 1, 2) == c2
@test AbstractNeuralNetworks.cell(g, 2, 1) == c3
@test AbstractNeuralNetworks.cell(g, 2, 2) == c4
@test AbstractNeuralNetworks.cell(g, 3, 1) == c5
@test AbstractNeuralNetworks.cell(g, 3, 2) == c6

@test_nowarn g(input, st, params)
