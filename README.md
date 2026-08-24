# Abstract Neural Networks

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/AbstractNeuralNetworks.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/AbstractNeuralNetworks.jl/latest/)
[![Build Status](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/AbstractNeuralNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/AbstractNeuralNetworks.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/A/AbstractNeuralNetworks.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/A/AbstractNeuralNetworks.html)

This package implements abstract and general data structures for the construction of neural networks, e.g., layers, chains, and architectures.
It mainly serves as a common base package for [GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl) and [SymbolicNetworks.jl](https://github.com/JuliaGNI/SymbolicNetworks.jl).


## Neural network parameters

The parameters of a network live in [NeuralNetworkParameters.jl](https://github.com/JuliaGNI/NeuralNetworkParameters.jl), as `NetworkParameters`, together with the tree walks, the flat form and the HDF5 path that go with them. `params(nn)` returns one:

```julia
using AbstractNeuralNetworks
using NeuralNetworkParameters: NetworkParameters

nn = NeuralNetwork(Chain(Dense(4, 3, tanh), Dense(3, 2, tanh)))
params(nn) isa NetworkParameters    # true
```

### Migrating from 0.6

Up to 0.6 this package defined and exported a struct of its own called `NeuralNetworkParameters`. As of 0.7 that name is gone from here — it is not aliased either, so that one type has one name across the ecosystem. Replace

```julia
using AbstractNeuralNetworks               # NeuralNetworkParameters came along
import AbstractNeuralNetworks: NeuralNetworkParameters
```

with

```julia
using NeuralNetworkParameters: NetworkParameters
```

and add `NeuralNetworkParameters` to your `Project.toml`. The type object is the same one, so `::Type{}` dispatch and `<:` bounds behave as they did. A set is built from keys and values with `NetworkParameters(NamedTuple{keys}(vals))`, since the braces of `NetworkParameters{T}` name its element type. `params`, `h5save`, `h5load`, `save` and `load` are still importable from `AbstractNeuralNetworks`.


## Development

We are using git hooks, e.g., to enforce that all tests pass before pushing. In order to activate these hooks, the following command must be executed once:
```
git config core.hooksPath .githooks
```