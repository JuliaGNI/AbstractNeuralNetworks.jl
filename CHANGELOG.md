# Changelog

## [Unreleased] — targeting 0.9.0

**Methods that `GeometricMachineLearning` (GML) defined on this package's types move here.** A
method on `Dense`, `Linear` or `NeuralNetwork` is type piracy wherever it is written outside this
package. GML wrote three in `src/layers/resnet.jl`, and the `save`/`load` methods in `ext/HDF5Ext.jl`.

Breaking for GML: a GML that still defines the three `resnet.jl` methods fails to precompile
against this release, with "Method overwriting is not permitted during Module precompilation".
GML has to delete them in the same change that raises its `AbstractNeuralNetworks` bound.

- **`Dense`, `Linear` and `Affine` gain 3-tensor methods**, in `src/layers/dense.jl`,
  `src/layers/linear.jl` and `src/layers/affine.jl`. GML's `src/layers/resnet.jl:63-73` defined
  `(d::Dense{M,N,true})(x::AbstractArray{T,3}, ps::NamedTuple)`, the matching `Dense{M,N,false}`
  method, and `(d::Linear{M,N})(...)`. The `Affine` method is new: without it, an `Affine` on a
  3-tensor is ambiguous between `(::Affine)(::AbstractArray, …)` and the `Dense{M,N,true}`
  3-tensor method. All four use one private helper:
  ```julia
  function _mul(W::AbstractMatrix, x::AbstractArray)
      reshape(W * reshape(x, size(x, 1), :), size(W, 1), Base.tail(size(x))...)
  end
  ```
  one reshape and one GEMM against every slice of the tensor, rather than GML's old
  `mat_tensor_mul`, which ran a hand-written KernelAbstractions `@kernel`. Checked against a
  per-slice loop, in `Float32` and `Float64`, and on a `JLArray` standing in for a device — new
  `@testset`s in `test/layers/dense_layer_tests.jl`, `test/layers/linear_layer_tests.jl` and
  `test/layers/affine_layer_tests.jl`.

- **A new weak extension, `ext/HDF5Ext.jl`, on `HDF5`.** GML's own `ext/HDF5Ext.jl` used to define
  `save(::HDF5.H5DataStore, ::NeuralNetwork)`, `save(::AbstractString, ::NeuralNetwork)`, and four
  `load(::Type{NeuralNetwork}, ...)` methods — `save`/`load` are `NeuralNetworkParameters`'
  functions, `NeuralNetwork` is this package's type, and `H5DataStore` is HDF5's, so none of the
  three names belonged to GML. The same methods, unchanged in behaviour, now live in this
  extension. `Project.toml` gains `HDF5` as a `[weakdeps]` entry with `HDF5Ext = "HDF5"` under
  `[extensions]`; `HDF5` stays in `[extras]` as before, and `JLArrays` joins it there for the
  device-backed 3-tensor tests. New `test/hdf5_tests.jl` covers a save/load round trip in both
  `Float32` and `Float64`, through an already-open `H5DataStore`, against a prototype parameter set.

## [0.8.0] — 2026-08-29

**A whole set of parameters is a `NetworkParameters`.** `NeuralNetworkParameters` 0.3.0 removes
`ParameterSet`, the `Union{NetworkParameters, NamedTuple}` these signatures were written on, because a
method on it is a method on `Base.NamedTuple` and because it gave one name to two questions — a whole
set, and a *branch* of one. Breaking: `applychain`, the `NetworkLoss`/`FeedForwardLoss` functors,
`_compute_loss`, `_statify`, `update!(::Model, …)` and `changebackend` take `NetworkParameters`.

A caller holding a bare `NamedTuple` writes `NetworkParameters(ps)`, which shares the leaf arrays
rather than copying them. `initialparameters` already returns one, so a network built here is
unaffected.

Two methods keep a `NamedTuple` signature, each for a stated reason rather than for breadth:

- **`applychain(layers, x, ps::NamedTuple)`**, because a reverse pass calls it that way.
  `NeuralNetworkParameters`' `ZygoteRules.pullback` for a `NetworkParameters` seeds the reverse pass
  with the *wrapped* `NamedTuple`, since that is what yields a tangent keyed by the layers rather than
  a tangent for the wrapper's one field. `test/custom_pullback_test.jl` fails without it.
- **`changebackend(backend, ::NamedTuple)`**, because moving a single *layer* between backends is a
  thing a caller legitimately asks for, and a layer is a branch rather than a set.

Both are written as separate methods rather than as one signature over a union: they answer different
questions that happen to share a body, and writing them out says which shape each caller is in.

The **loss functors take `ps` untyped** rather than a method per shape, and that difference from
`applychain` is deliberate: `applychain`'s two methods *normalise* — they name the two shapes `values`
is defined on before handing a `Tuple` to the `@generated` method that does the work — while the loss
functors forward `ps` untouched to `model(input, ps)` and read nothing of it. What it costs is that a
wrong `ps` fails inside `model(input, ps)` rather than at the call.

### Renamed

- **`ArrayNamedTuple` is `NamedTupleOfArrays`**, and `ArrayTuple` is `TupleOfArrays`. The name says
  what the alias is about, which here is a network's *inputs and outputs*; `GeometricOptimizers` had an
  `ArrayNamedTuple` of *parameters*, and the two shared a name by coincidence rather than by meaning.
  That one is gone as of `GeometricOptimizers` 0.7.0, so this is the only alias of the shape left in
  the ecosystem. `ArrayOrNamedTuple` keeps its name.
