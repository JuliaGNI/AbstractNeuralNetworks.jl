# Static Neural Network Parameters

We can also allocate neural network parameters using [`StaticArrays`](https://github.com/JuliaArrays/StaticArrays.jl). Therefore we simply need to set the keyword `static` to true in the [`NeuralNetwork`](@ref) constructor. 

!!! warning
    Static neural network parameters are only supported for dense CPU arrays.

```@example static_parameters
using AbstractNeuralNetworks
import Random
Random.seed!(123)

backend = AbstractNeuralNetworks.CPUStatic()
c = Chain(Dense(2, 10, tanh), Dense(10, 1, tanh))
nn = NeuralNetwork(c, backend)
typeof(nn.params.L1.W)
```

We can compare different evaluation times:
```@example
nn_cpu = changebackend(CPU(), nn)
```