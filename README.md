# Abstract Neural Networks

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/AbstractNeuralNetworks.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/AbstractNeuralNetworks.jl/latest/)
[![Build Status](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/AbstractNeuralNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/AbstractNeuralNetworks.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/A/AbstractNeuralNetworks.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/A/AbstractNeuralNetworks.html)

This package implements abstract and general data structures for the construction of neural networks, e.g., layers, chains, and architectures.
It mainly serves as a common base package for [GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl) and [SymbolicNetworks.jl](https://github.com/JuliaGNI/SymbolicNetworks.jl).


## Development

We are using git hooks, e.g., to enforce that all tests pass before pushing. In order to activate these hooks, the following command must be executed once:
```
git config core.hooksPath .githooks
```