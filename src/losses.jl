@doc raw"""
    NetworkLoss

An abstract type for all the neural network losses. 
If you want to implement `CustomLoss <: NetworkLoss` you need to define a functor:
```julia
(loss::CustomLoss)(model, ps, input, output)
```
where `model` is an instance of an `AbstractExplicitLayer` or a `Chain` and `ps` the parameters.

See [`FeedForwardLoss`](@ref), `GeometricMachineLearning.TransformerLoss`, `GeometricMachineLearning.AutoEncoderLoss` and `GeometricMachineLearning.ReducedLoss` for examples.
"""
abstract type NetworkLoss end

@doc raw"""
    FeedForwardLoss()

Make an instance of a loss for feedforward neural networks.

This should be used together with a neural network of type `GeometricMachineLearning.NeuralNetworkIntegrator`.

# Example 

`FeedForwardLoss` applies a neural network to an input and compares it to the `output` via an ``L_2`` norm:

```jldoctest 
using AbstractNeuralNetworks
using LinearAlgebra: norm
import Random
Random.seed!(123)

const d = 2
arch = Chain(Dense(d, d), Dense(d, d))
nn = NeuralNetwork(arch)

input_vec =  [1., 2.]
output_vec = [3., 4.]
loss = FeedForwardLoss()

loss(nn, input_vec, output_vec) ≈ norm(output_vec - nn(input_vec)) / norm(output_vec)

# output

true
```

So `FeedForwardLoss` simply does:

```math
    \mathtt{loss}(\mathcal{NN}, \mathtt{input}, \mathtt{output}) = || \mathcal{NN}(\mathtt{input}) - \mathtt{output} || / || \mathtt{output}||,
```
where ``||\cdot||`` is the ``L_2`` norm. 

# Parameters

This loss does not have any parameters.
"""
struct FeedForwardLoss <: NetworkLoss end

# `map` over `values(...)` rather than `NamedTuple{keys}(generator)`: the generator collects to a
# `Vector`, and `ChainRulesCore.ProjectTo` has no method for a `Vector` of arrays as the tangent of
# a `NamedTuple`, so any gradient through it throws. Mapping tuple-to-tuple keeps the tangent a
# `Tuple` and differentiates. This used to be worked around by hardcoding the `(:q, :p)` case in
# `_norm`/`_diff` below, which left every other `NamedTuple` undifferentiable.
function apply(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(map(fun, map(values, ps)...))
end

# overload norm
_norm(dx::NamedTuple) = sum(map(_norm, values(dx))) / √length(dx)
_norm(A::AbstractArray) = norm(A)

# overloaded +/- operation 
_diff(dx₁::NamedTuple, dx₂::NamedTuple) = apply(_diff, dx₁, dx₂)
_diff(A::AbstractArray, B::AbstractArray) = A - B
_add(dx₁::NamedTuple, dx₂::NamedTuple) = apply(_add, dx₁, dx₂)
_add(A::AbstractArray, B::AbstractArray) = A + B

function (loss::NetworkLoss)(nn::NeuralNetwork, input::ArrayOrNamedTuple, output::ArrayOrNamedTuple)
    loss(nn.model, nn.params, input, output)
end

function _compute_loss(output_prediction::ArrayOrNamedTuple, output::ArrayOrNamedTuple)
    _norm(_diff(output_prediction, output)) / _norm(output)
end

# `ps` is untyped here and in the two functors below. Both shapes reach them: a whole set of parameters
# is a `NetworkParameters`, and the bare `NamedTuple` that a *reverse pass* produces arrives too,
# because `NeuralNetworkParameters`' `ZygoteRules.pullback` for a container seeds the pass with the
# wrapped `NamedTuple` rather than the container. Nothing here is dispatched on either — the model and
# the input/output types settle every method — so naming a type would only be a claim about which of
# the two shapes is allowed, and both are.
#
# **`applychain` writes both out and this does not, and that is not an inconsistency.** These forward
# `ps` untouched to `model(input, ps)`; `applychain` *normalises* it, and its two methods exist to name
# the two shapes `values` is defined on before handing a `Tuple` to the `@generated` method that does
# the work. There the shape is the subject of the method; here it passes through.
#
# The cost of leaving it untyped is that a wrong `ps` fails inside `model(input, ps)` rather than at
# the call. That is accepted rather than overlooked: an annotation naming both shapes would be a
# `Union` over `Base.NamedTuple`, and two methods per functor would put six signatures in this file to
# assert something no method reads.
function _compute_loss(model::Union{AbstractExplicitLayer, Chain}, ps,
        input::ArrayOrNamedTuple, output::ArrayOrNamedTuple)
    output_prediction = model(input, ps)
    _compute_loss(output_prediction, output)
end

function (loss::NetworkLoss)(model::Union{Chain, AbstractExplicitLayer}, ps,
        input::ArrayOrNamedTuple, output::ArrayOrNamedTuple)
    error("Functor not defined for `NetworkLoss` of type $(typeof(loss)).")
end

function (loss::FeedForwardLoss)(model::Union{Chain, AbstractExplicitLayer}, ps,
        input::ArrayOrNamedTuple, output::ArrayOrNamedTuple)
    _compute_loss(model, ps, input, output)
end
