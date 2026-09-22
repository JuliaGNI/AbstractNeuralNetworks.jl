# Changelog

## [Unreleased] — targeting 0.9.0

**Three methods `GeometricMachineLearning` (GML) defined on ANN's own types move here**, because a
method on `AbstractNeuralNetworks`' `Dense`, `Linear` or `NeuralNetwork`, or on `HDF5.H5DataStore`,
is type piracy wherever it is written outside this package — GML committed it, in its
`ext`-less `src/layers/resnet.jl` and in `ext/HDF5Ext.jl`. Nothing about behaviour changes here,
only which package defines it.

- **`Dense` and `Linear` gain their own 3-tensor methods**, in `src/layers/dense.jl` and
  `src/layers/linear.jl`. GML's `src/layers/resnet.jl:63-73` used to define
  `(d::Dense{M,N,true})(x::AbstractArray{T,3}, ps::NamedTuple)`, the matching `Dense{M,N,false}`
  method, and `(d::Linear{M,N})(...)` — three methods on ANN's own types. All three now live here,
  built on one private helper:
  ```julia
  function _mul(W::AbstractMatrix, x::AbstractArray)
      reshape(W * reshape(x, size(x, 1), :), size(W, 1), Base.tail(size(x))...)
  end
  ```
  one reshape and one GEMM against every slice of the tensor, rather than GML's old
  `mat_tensor_mul`, which ran a hand-written KernelAbstractions `@kernel`. The name and the product
  are deliberately the same `_mul` that GML's own `M3` task-file part will introduce elsewhere, so
  GML's later kernels already match what is defined here. Checked against a per-slice loop, in
  `Float32` and `Float64`, and on a `JLArray` standing in for a device — new `@testset`s in
  `test/layers/dense_layer_tests.jl` and `test/layers/linear_layer_tests.jl`.

- **A new weak extension, `ext/HDF5Ext.jl`, on `HDF5`.** GML's own `ext/HDF5Ext.jl` used to define
  `save(::HDF5.H5DataStore, ::NeuralNetwork)`, `save(::AbstractString, ::NeuralNetwork)`, and three
  `load(::Type{NeuralNetwork}, ...)` methods — `save`/`load` are `NeuralNetworkParameters`'
  functions, `NeuralNetwork` is this package's type, and `H5DataStore` is HDF5's, so none of the
  three names belonged to GML. The same methods, unchanged in behaviour, now live in this
  extension. `Project.toml` gains `HDF5` as a `[weakdeps]` entry with `HDF5Ext = "HDF5"` under
  `[extensions]`; `HDF5` stays in `[extras]` as before, and `JLArrays` joins it there for the
  device-backed 3-tensor tests. New `test/hdf5_tests.jl` covers a save/load round trip in both
  `Float32` and `Float64`, through an already-open `H5DataStore`, against a prototype parameter set.

Both sets of methods were previously reachable only by loading GML, which is why they read as
GML's code. `GeometricMachineLearning`'s own `src/layers/resnet.jl` and `ext/HDF5Ext.jl` still need
the corresponding deletions once it can depend on this release — that is a separate, later change
in GML, not this one. `Aqua.test_piracies(GeometricMachineLearning)` was checked against a scratch
copy of GML with those three `resnet.jl` methods and the `HDF5Ext.jl` content removed (not a
committed GML change) and reported zero piracies, confirming the fix is complete once GML catches
up.

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
