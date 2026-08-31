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

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.