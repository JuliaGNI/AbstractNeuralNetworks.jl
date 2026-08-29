# Changelog

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
