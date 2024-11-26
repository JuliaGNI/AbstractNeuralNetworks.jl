@doc raw"""
    NetworkLoss

An abstract type for all the neural network losses. 
If you want to implement `CustomLoss <: NetworkLoss` you need to define a functor:
```julia
(loss::CustomLoss)(model, ps, input, output)
```
where `model` is an instance of an `AbstractExplicitLayer` or a `Chain` and `ps` the parameters.

See [`FeedForwardLoss`](@ref), [`TransformerLoss`](@ref), [`AutoEncoderLoss`](@ref) and [`ReducedLoss`](@ref) for examples.
"""
abstract type NetworkLoss end 

const QPT{T} = NamedTuple{(:q, :p), Tuple{AT, AT}} where {T, AT <: AbstractArray{T}}
const QPTOAT{T} = Union{QPT{T}, AbstractArray{T}} where T

function (loss::NetworkLoss)(nn::NeuralNetwork, input::QPTOAT, output::QPTOAT)
    loss(nn.model, nn.params, input, output)
end

function _compute_loss(output_prediction::QPTOAT, output::QPTOAT)
    _norm(_diff(output_prediction, output)) / _norm(output)
end 

function _compute_loss(model::Union{AbstractExplicitLayer, Chain}, ps::Union{NeuralNetworkParameters, NamedTuple}, input::QPTOAT, output::QPTOAT)
    output_prediction = model(input, ps)
    _compute_loss(output_prediction, output)
end

function (loss::NetworkLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{NeuralNetworkParameters, NamedTuple}, input::QPTOAT, output::QPTOAT)
    _compute_loss(model, ps, input, output)
end

@doc raw"""
    FeedForwardLoss()

Make an instance of a loss for feedforward neural networks.

This should be used together with a neural network of type `GeometricMachineLearning.NeuralNetworkIntegrator`.

# Example 

`FeedForwardLoss` applies a neural network to an input and compares it to the `output` via an ``L_2`` norm:

```jldoctest 
using GeometricMachineLearning
using LinearAlgebra: norm
import Random
Random.seed!(123)

const d = 2
arch = GSympNet(d)
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