function changebackend(backend::Backend, x::AbstractArray{T}) where T
    _x = KernelAbstractions.allocate(backend, T, size(x)...)
    KernelAbstractions.copyto!(backend, _x, x)
    nothing
end

# this is pretty ugly
function changebackend(backend::Backend, x::MArray)
    changebackend(backend, Array(x))
end

function changebackend(backend::Backend, ps::NamedTuple)
    ps_vals = Tuple(changebackend(backend, x) for x in values(ps))
    NamedTuple{keys(ps)}(ps_vals)
end

function changebackend(backend::Backend, ps::NeuralNetworkParameters)
    NeuralNetworkParameters(changebackend(backend, ps.params))
end

"""
    changebackend(backend, nn)


# Extended help

The function `changebackend` is defined for [`NeuralNetworkParameters`](@ref), [`NeuralNetwork`](@ref), `AbstractArray`s and `NamedTuple`s. This function is also exported.
"""
function changebackend(backend::Backend, nn::NeuralNetwork)
    NeuralNetwork(nn.architecture, nn.model, changebackend(backend, nn.params), backend)
end