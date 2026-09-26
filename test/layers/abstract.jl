using AbstractNeuralNetworks
using Test

# `AbstractLayer{M, N}` is a map from R^M to R^N, and `input_dimension`/`output_dimension` read the
# first and second parameter respectively. This testset was empty, so nothing pinned that order.

@test input_dimension(Dense(4, 5, tanh)) == 4
@test output_dimension(Dense(4, 5, tanh)) == 5

@test input_dimension(Linear(4, 5)) == 4
@test output_dimension(Linear(4, 5)) == 5

@test input_dimension(Affine(4, 5)) == 4
@test output_dimension(Affine(4, 5)) == 5

@test Dense(4, 5, tanh) isa AbstractNeuralNetworks.AbstractLayer{4, 5}
