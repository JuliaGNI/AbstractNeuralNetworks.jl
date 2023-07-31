
abstract type AbstractInitializer end

const Initializer = Union{AbstractInitializer, Base.Callable}

struct ZeroInitializer <: AbstractInitializer end
function (::ZeroInitializer)(_, x) 
    x .= KernelAbstractions.zero(x)
end

struct OneInitializer <: AbstractInitializer end
function (::OneInitializer)(_, x::AbstractArray{T}) where T 
    backend = get_backend(x)
    x .= KernelAbstractions.ones(backend, T, size(x))
end

default_initializer() = randn!

struct GlorotUniform <: AbstractNeuralNetworks.AbstractInitializer end

function (::GlorotUniform)(rng, x::AbstractVecOrMat{T}) where T
    rand!(rng, x)
    x .= sqrt(T(24.0) / sum(size(x))) * (x .- T(0.5)) 
end