# Static Neural Network Parameters

We can also allocate neural network parameters using [`StaticArrays`](https://github.com/JuliaArrays/StaticArrays.jl). Therefore we simply need to set the keyword `static` to true in the [`NeuralNetwork`](@ref) constructor. 

!!! warning
    Static neural network parameters are only supported for dense CPU arrays. `AbstractNeuralNetworks` defines a type `CPUStatic`, but does not have equivalent GPU objects.

```@example static_parameters
using AbstractNeuralNetworks
import Random
Random.seed!(123)

backend = AbstractNeuralNetworks.CPUStatic()
input_dim = 2
n_hidden_layers = 100
c = Chain(Dense(input_dim, 10, tanh), Tuple(Dense(10, 10, tanh) for _ in 1:n_hidden_layers)..., Dense(10, 1, tanh))
nn = NeuralNetwork(c, backend)
typeof(params(nn).L1.W)
```

We can compare different evaluation times:
```@example static_parameters
nn_cpu = changebackend(CPU(), nn)
second_dim = 200
x = rand(input_dim, second_dim)
nn(x); # hide
@time nn(x);
nothing # hide
```

```@example static_parameters
nn_cpu(x); # hide
@time nn_cpu(x);
nothing # hide
```

If we also make the *input* static, we get:

```@example static_parameters
using StaticArrays
x = @SMatrix rand(input_dim, second_dim)
nn(x);
@time nn(x);
nothing # hide
```

```@example static_parameters
nn_cpu(x); # hide
@time nn_cpu(x);
nothing # hide
```