using Aqua
using AbstractNeuralNetworks
using Test

Aqua.test_all(AbstractNeuralNetworks; ambiguities = false)

# two ambiguities of `apply`, issue #43
Aqua.test_ambiguities(AbstractNeuralNetworks; broken = true)
