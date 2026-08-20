@doc raw"""
    NeuralNetworkParameters

Compatibility alias for `NeuralNetworkParameters.NetworkParameters`, which is where the parameters of
a neural network live as of 0.7.0.

The type is upstream now because the traversal that goes with it — flattening, saving, mapping a
backend change over the leaves — was being written once per package. It is called
`NetworkParameters` there because a package cannot export a type sharing its own name: the module
binding wins at the `using` site, so `NeuralNetworkParameters(nt)` would try to call a `Module`.

This alias is **not exported**. Reach it with

```julia
import AbstractNeuralNetworks: NeuralNetworkParameters
```

or, preferably in new code, use `NetworkParameters` from `NeuralNetworkParameters` directly. It is the
same object either way, so `::Type{}` dispatch, `<: NeuralNetworkParameters` bounds and
`NeuralNetworkParameters{keys}(vals)` construction all behave as they did.
"""
const NeuralNetworkParameters = NetworkParameters
