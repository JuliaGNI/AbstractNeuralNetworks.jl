# Changelog

## [0.8.0]

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

