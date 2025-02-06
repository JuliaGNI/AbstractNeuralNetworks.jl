function changebackend(backend::NeuralNetworkBackend, x::AbstractArray{T}) where T
    _x = KernelAbstractions.allocate(backend, T, size(x)...)
    KernelAbstractions.copyto!(backend, _x, x)
    _x
end

# this is pretty ugly
function changebackend(backend::NeuralNetworkBackend, x::MArray)
    changebackend(backend, Array(x))
end

function changebackend(backend::NeuralNetworkBackend, ps::NamedTuple)
    ps_vals = Tuple(changebackend(backend, x) for x in values(ps))
    NamedTuple{keys(ps)}(ps_vals)
end

function changebackend(backend::NeuralNetworkBackend, ps::NeuralNetworkParameters)
    NeuralNetworkParameters(changebackend(backend, params(ps)))
end

"""
    changebackend(backend, nn)

# Extended help

The function `changebackend` is defined for [`NeuralNetworkParameters`](@ref), [`NeuralNetwork`](@ref), `AbstractArray`s and `NamedTuple`s. This function is also exported.
"""
function changebackend(backend::NeuralNetworkBackend, nn::NeuralNetwork)
    NeuralNetwork(architecture(nn), model(nn), changebackend(backend, params(nn)), backend)
end