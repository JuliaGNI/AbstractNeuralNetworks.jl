using Zygote
using LinearAlgebra: norm
using AbstractNeuralNetworks
using AbstractNeuralNetworks: applychain
using Test

nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))

gradient_result = Zygote.gradient(ps -> norm(applychain(nn.model.layers, rand(10), ps)), nn.params)[1]
pullback_result = Zygote.pullback(applychain, nn.model.layers, rand(10), nn.params)[2](rand(10))[3]

@test typeof(gradient_result) <: NeuralNetworkParameters 
@test typeof(pullback_result) <: NeuralNetworkParameters