function changebackend(backend::NeuralNetworkBackend, x::AbstractArray{T}) where {T}
    _x = KernelAbstractions.allocate(backend, T, size(x)...)
    KernelAbstractions.copyto!(backend, _x, x)
    _x
end

# this is pretty ugly
function changebackend(backend::NeuralNetworkBackend, x::MArray)
    changebackend(backend, Array(x))
end

# `mapparameters` recurses through the branches and hands `f` the leaves, rebuilding what it was given.
# It descends into `Tuple` branches too, so a multi-block leaf is covered without a method of its own.
#
# Two methods and not one on a union of the two types: a whole set of parameters is a
# `NetworkParameters`, and a bare `NamedTuple` reaching here is a *branch* of one — a single layer,
# which is a thing a caller legitimately asks to move on its own. They are different questions that
# happen to share a body, so they are written as what they are. `changebackend` is this package's own
# function, so a method on `NamedTuple` owns its signature.
function changebackend(backend::NeuralNetworkBackend, ps::NetworkParameters)
    mapparameters(x -> changebackend(backend, x), ps)
end

function changebackend(backend::NeuralNetworkBackend, layer::NamedTuple)
    mapparameters(x -> changebackend(backend, x), layer)
end

"""
    changebackend(backend, nn)

# Extended help

The function `changebackend` is defined for [`NeuralNetwork`](@ref), `AbstractArray`s, and the
`NamedTuple`s and `NetworkParameters` of a parameter set — `Tuple` branches inside such a set are
descended into as well. This function is also exported.
"""
function changebackend(backend::NeuralNetworkBackend, nn::NeuralNetwork)
    NeuralNetwork(architecture(nn), model(nn), changebackend(backend, params(nn)), backend)
end
