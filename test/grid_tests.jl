using AbstractNeuralNetworks
using Random
using Test

import AbstractNeuralNetworks: update!

rng = Random.seed!(1234)

input = [[2,2],[2]]
st = [[3,4], [7,8,9], [1, 2, 3, 4, 5, 6, 7, 8]]

c1 = Recurrent(2, 2, 2, 2, tanh)
c2 = Recurrent(1, 2, 1, 1, tanh)
c3 = Recurrent(2, 3, 2, 2, tanh)
c4 = IdentityCell()
c5 = GRU(2, 8)
c6 = LSTM(1, 4)

p1 = initialparameters(rng, c1, Float64)
p3 = initialparameters(rng, c3, Float64)
p5 = initialparameters(rng, c5, Float64)
p2 = initialparameters(rng, c2, Float64)
p4 = initialparameters(rng, c4, Float64)
p6 = initialparameters(rng, c6, Float64)

params = [p1 p2 ; p3 p4 ; p5 p6]

g = GridCell( [c1  c2;
               c3  c4;
               c5  c6])

@test length(g) == 6
@test_nowarn for c in g end
@test Tuple([e for e in eachindex(g)]) == ((1,1),(2,1), (3,1), (1,2), (2,2), (3,2))

rng = Random.seed!(1234)
@test params == initialparameters(rng, g, Float64)

@test AbstractNeuralNetworks.cell(g, 1, 1) == c1
@test AbstractNeuralNetworks.cell(g, 1, 2) == c2
@test AbstractNeuralNetworks.cell(g, 2, 1) == c3
@test AbstractNeuralNetworks.cell(g, 2, 2) == c4
@test AbstractNeuralNetworks.cell(g, 3, 1) == c5
@test AbstractNeuralNetworks.cell(g, 3, 2) == c6

@test AbstractNeuralNetworks.cell(g, 1) == c1
@test AbstractNeuralNetworks.cell(g, 2) == c3
@test AbstractNeuralNetworks.cell(g, 3) == c5
@test AbstractNeuralNetworks.cell(g, 4) == c2
@test AbstractNeuralNetworks.cell(g, 5) == c4
@test AbstractNeuralNetworks.cell(g, 6) == c6

@test_nowarn g(input, st, params)

@test_nowarn update!(g, params, params, 0.4)
