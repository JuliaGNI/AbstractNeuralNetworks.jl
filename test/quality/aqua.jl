using Aqua
using AbstractNeuralNetworks
using Test

Aqua.test_all(AbstractNeuralNetworks; ambiguities = false)

# two ambiguities of `apply`
Aqua.test_ambiguities(AbstractNeuralNetworks; broken = true) # issue #43
