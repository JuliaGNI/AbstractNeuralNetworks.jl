# Changelog

## [0.10.0] — 2026-11-09

### Breaking Changes

- **Requires KernelAbstractions 0.10** (`KernelAbstractions = "0.10.5"`; 0.9 is no longer
  accepted). An environment that holds KernelAbstractions 0.9 cannot install this release. On
  0.10, `CPU()` is an alias of the OpenCL backend `POCLBackend`, which runs on PoCL. This package
  launches no kernel, so nothing here runs differently: `zeros`, `ones` and `allocate` on `CPU()`
  still return an `Array`, and `NeuralNetwork(model, CPU())` still defaults to `Float64`.
  The floor is 0.10.5 because the test dependency JLArrays 0.5, the first release that accepts
  KernelAbstractions 0.10, requires it. CUDA 6.5 and later cannot be installed beside
  KernelAbstractions 0.9, and the backends of Metal 1.12 and CUDA 6.5 subtype
  `KernelAbstractions.Backend` only on 0.10, so only there are they a `NeuralNetworkBackend`.

- **`NeuralNetwork(model, backend)` defaults to `Float32` on every KernelAbstractions backend other
  than `CPU()`.** The method now dispatches on `KernelAbstractions.Backend`, because 0.10 deprecates
  `GPU` and makes it an alias of `Backend`. On 0.9, a backend that was not a subtype of `GPU`
  raised "Default type for … not defined"; that method is removed, since no backend reaches it
  any more. `CPU()` and `CPUStatic()` still default to `Float64`.

- **`GPU` is no longer exported**; the package re-exported it from KernelAbstractions. After
  `using AbstractNeuralNetworks` alone, the bare `GPU` raises `UndefVarError`. `CPU` stays
  exported. Code that still needs the name writes `KernelAbstractions.GPU`, which prints a
  deprecation warning, or uses `KernelAbstractions.Backend`. On KernelAbstractions 0.10, `GPU` is
  an alias of `Backend`, so `CPU() isa GPU` is true. This is why the export is dropped: a
  re-exported `GPU` printed no deprecation warning, so code that dispatches on it changed meaning
  without notice. A package that re-exports `GPU` through this one, as GeometricMachineLearning
  does, has to drop that export when it moves to this release.

### Changed

- **The `[compat]` lower bounds resolve together, and the tests pass at them on Julia 1.12.**
  The CI Downgrade job resolves `Project.toml` and `test/Project.toml` together at their lower
  bounds and runs the tests there. Three floors are raised for this:
  - `GPUArraysCore` from 0.2.0 to **0.2.1**. JLArrays 0.5 needs GPUArrays 12.1 or later, and
    GPUArrays 12.1 requires GPUArraysCore 0.2.1 in the General registry.
  - `Zygote` (tests) from `0.6, 0.7` to **0.7.5**, the first release after 0.6.41 that does not
    depend on GPUArrays. Zygote 0.6.42 through 0.7.4 cap it at 11 or lower and JLArrays 0.5 needs
    12.1 or later, so none of them resolves beside it. Zygote 0.6.41 and earlier resolve, but the
    merged resolve picks 0.6.35, which fails on Julia 1.12 in `has_chain_rrule`. The tests no
    longer accept Zygote 0.6.
  - `Aqua` (tests) gets a compat entry, **0.8.14**. Without one the merged resolve selects Aqua
    0.1.0. Aqua 0.8.14 is the first release whose piracy check handles the method table of Julia
    1.12; before it, the check fails because `Core.TypeName` has no field `mt`.

- **The tests use JLArrays 0.5** (`test/Project.toml`, from 0.3), the first JLArrays release that
  accepts KernelAbstractions 0.10. The `JLArray` cases of the layer tests stay.

- **CI runs the tests with `check_bounds: auto`**, from the action's default `yes`, in the test
  matrix and in the Downgrade job. The workflow is the shared CI workflow of the package family,
  whose compiled kernels on the PoCL `CPU()` backend cannot allocate under forced bounds checks.
  This package launches no kernel; its suite passes in both modes.

## [0.9.3] — 2026-10-24

### Changed

- **Julia 1.12 is the floor** (`[compat] julia = "1.12"`, from 1.11). A patch release, because no
  API changes: a caller on Julia 1.11 can no longer install this version.
  Nothing in the package's code changes: it held no version-conditional code and no workaround
  for the older floor, so there was nothing to delete.

## [0.9.2] — 2026-10-23

### New Features

- **`NamedTupleOfArrays` and `ArrayOrNamedTuple` are `public`.** A downstream package can now
  write them in its method signatures as part of this package's API, in place of its own
  `(:q, :p)` aliases. They stay unexported, so `using AbstractNeuralNetworks` brings no new name
  into scope; import them as `using AbstractNeuralNetworks: NamedTupleOfArrays, ArrayOrNamedTuple`.
  `TupleOfArrays` stays internal.

## [0.9.1] — 2026-10-11

### Changed

- **CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of `Julia min`, and a test
  job saves the Julia cache only when it succeeds.**

- **`Pkg.test()` no longer runs the doctests.** `test/quality/doctests.jl` is now the group
  `doctests`, which an empty `ARGS` does not run; `Pkg.test(test_args = ["doctests"])` runs it. In
  CI the Doctests job stays their runner, so the test matrix no longer runs them a second time.

- **`test/neural_network_constructors.jl` moves to `test/integration/neural_network_constructors.jl`.**
  The test convention keeps a test file at the top level of `test/` only where it mirrors
  `src/<name>.jl`, and there is no `src/neural_network_constructors.jl`. The file tests the
  `NeuralNetwork` constructors of `src/neural_network.jl`, whose mirror `test/neural_network.jl`
  already exists, so it goes to `test/integration/`.

### Dependencies and tests

- **Admits `NeuralNetworkParameters` 0.5** alongside 0.4.2 (compat only; no change to this
  package's own behaviour). The downstream packages move to NNP 0.5, and they can resolve only if
  this package admits it. NNP 0.5 gives an untouched leaf a zero gradient instead of `nothing`,
  gives a leaf's gradient the leaf's element type, rejects a `NetworkParameters` with two numeric
  element types, and makes `+` add storage leaf by leaf. The test suite passes against both 0.4.2
  and 0.5.

## [0.9.0] — 2026-10-02

**Methods that `GeometricMachineLearning` (GML) defined on this package's types move here.** A
method on `Dense`, `Linear` or `NeuralNetwork` is type piracy wherever it is written outside this
package. GML defined three such methods, the 3-tensor methods on `Dense` and `Linear`, and the
`save`/`load` methods for `NeuralNetwork` in its HDF5 extension.

### Breaking Changes

- **A GML that still defines the three 3-tensor methods fails to precompile** against this
  release, with "Method overwriting is not permitted during Module precompilation". GML has to
  delete them in the same change that raises its `AbstractNeuralNetworks` bound.

- **The `backend` keyword of `load(NeuralNetwork, …)` is removed**; the network loads on the CPU
  and `changebackend(backend, nn)` moves it to a device with parameters included.

### New Features

- **`Dense`, `Linear` and `Affine` accept inputs of rank 3 or more.** Each layer applies its
  weight matrix to the first dimension, preserving the remaining dimensions. The implementation
  uses only `reshape` and matrix multiplication, so it runs on any array type that supports both
  — including device-backed arrays.

- **A new weak extension, `HDF5Ext`, provides `save` and `load` for `NeuralNetwork`.** The
  extension defines methods for HDF5 stores; filename forms are `NeuralNetworkParameters`' own
  methods. The extension loads automatically when HDF5 is available.

### Dependencies and tests

- **Requires `NeuralNetworkParameters` 0.4** (compat only; no change to this package's own
  behaviour). Under Zygote, NNP 0.4 calls the loss with the `NetworkParameters` itself rather than
  the wrapped `NamedTuple`, and converts the cotangent of each leaf with `storage_gradient` to the
  gradient with respect to that leaf's storage. A `Chain` still accepts a whole set as a bare
  `NamedTuple` as well: `SymbolicNeuralNetworks` builds symbolic networks whose parameters are one.

- **Requires `NeuralNetworkParameters` 0.4.2** (compat only; no change to this package's own
  behaviour). NNP 0.4.2 requires GeometricBase 0.15, which declares its stubs public and requires
  Julia 1.11; this package's Julia floor is 1.11 already.

- **The test suite follows the common layout** (no change to the package's behaviour). The test
  dependencies move from `[extras]`/`[targets]` to `test/Project.toml`; Aqua and Documenter are
  new test dependencies. `runtests.jl` runs the groups `core` and `slow`, and each test file
  sits at the path of the source file it tests. New: `test/quality/aqua.jl`, with the check for
  method ambiguities marked broken for the two ambiguities of `apply` (#43), and
  `test/quality/doctests.jl` in `slow`. `LinearAlgebra` and `Random` get the compat bound `"1"`,
  which Aqua requires. The empty `test/architecture_tests.jl`, which held no test, is removed.

- **`test/Project.toml` no longer repeats the root's bounds** (test environment only; no change
  to the package). Its `[compat]` table loses the entries for `HDF5`, `KernelAbstractions`,
  `LinearAlgebra`, `NeuralNetworkParameters`, `Random` and `StaticArrays`, which are dependencies
  or weak dependencies of the root `Project.toml`. The rule: a test or docs environment carries a
  `[compat]` entry only for a dependency that the root does not declare, so the root's bound
  alone governs the tests. This removes the `"1"` bounds for `LinearAlgebra` and `Random` of the
  entry above; the root keeps its own.

### Bug Fixes

- **Bug fix: a top-level `Zygote.pullback(applychain, layers, x, ps)` gives the storage gradient
  of each leaf.** It returned the natural cotangent of each leaf in a `NetworkParameters`, which
  NNP 0.4 reads as the gradient with respect to the storage. For a leaf whose `storage_gradient` is
  not the identity, that gradient was wrong and came back as a dense matrix. It now converts each
  leaf with NNP's `map_cotangent(storage_gradient, …)` and keeps the leaf's type; the return shape
  is unchanged. A loss through a `Chain` was not affected. It also handles the case where the
  parameter cotangent is `nothing` (when all layers have no parameters), which previously raised
  a `MethodError`. Raises the `NeuralNetworkParameters` bound to 0.4.1, the release that makes
  `map_cotangent` public.

- **`parameterlength(::Dense)` reads the bias flag as a `Bool`**, as the in-place call
  `(::Dense)(y, x, ps)` already does, instead of comparing it with `== true`. A `Dense` built with
  a non-`Bool` `use_bias`, such as `1`, now raises a `TypeError` in `parameterlength`; it returned
  the parameter count with the bias for `1` and `1.0`, and without it for `0`. Such a `Dense`
  already could not be used: `initialparameters` and the out-of-place call have methods only for
  `true` and `false`. The tests assert `usebias(l)` and `!usebias(l)` directly.

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
