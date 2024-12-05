"""
    NeuralNetworkBackend

The backend that specifies where and how neural network parameters are allocated.

It largely inherits properties from [`KernelAbstractions.Backend`](https://github.com/JuliaGPU/KernelAbstractions.jl), but also adds `CPUStatic` which is defined in `AbstractNeuralNetworks`.
"""
const NeuralNetworkBackend = Union{KernelAbstractions.Backend, CPUStatic}

function networkbackend(backend::NeuralNetworkBackend) 
    error("Function `networkbackend` not defined for $(backend)")
end

"""
    networkbackend(arr)

Returns the [`NeuralNetworkBAckend`](@ref) of `arr`.
"""
function networkbackend(arr::AbstractArray)
    KernelAbstractions.get_backend(arr)
end