function changebackend(backend::NeuralNetworkBackend, x::AbstractArray{T}) where T
    _x = KernelAbstractions.allocate(backend, T, size(x)...)
    KernelAbstractions.copyto!(backend, _x, x)
    _x
end

# this is pretty ugly
function changebackend(backend::NeuralNetworkBackend, x::MArray)
    changebackend(backend, Array(x))
end

# One walk covers both containers: `mapparameters` recurses through the `NamedTuple`s and hands `f`
# the leaves, returning a `NeuralNetworkParameters` for a `NeuralNetworkParameters` and a
# `NamedTuple` for a `NamedTuple`. It also descends into `Tuple` branches, which the two methods this
# replaces did not — they handed a `Tuple` straight to `KernelAbstractions.allocate`.
function changebackend(backend::NeuralNetworkBackend, ps::Union{NamedTuple, NeuralNetworkParameters})
    mapparameters(x -> changebackend(backend, x), ps)
end

"""
    changebackend(backend, nn)

# Extended help

The function `changebackend` is defined for [`NeuralNetworkParameters`](@ref), [`NeuralNetwork`](@ref), `AbstractArray`s and `NamedTuple`s. This function is also exported.
"""
function changebackend(backend::NeuralNetworkBackend, nn::NeuralNetwork)
    NeuralNetwork(architecture(nn), model(nn), changebackend(backend, params(nn)), backend)
end