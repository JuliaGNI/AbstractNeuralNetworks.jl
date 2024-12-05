"""
    Initializer

Determines how neural network weights are initialized.
"""
abstract type Initializer end

"""
    ZeroInitializer <: Initializer
"""
struct ZeroInitializer <: Initializer end

function (::ZeroInitializer)(_, x) 
    x .= KernelAbstractions.zero(x)
    
    nothing
end

"""
    OneInitializer <: Initializer
"""
struct OneInitializer <: Initializer end

function (::OneInitializer)(_, x::AbstractArray{T}) where T 
    backend = networkbackend(x)
    x .= KernelAbstractions.ones(backend, T, size(x))

    nothing
end

"""
    GlorotUniform <: Initializer

Glorot uniform was introduced by [glorot2010understanding](@cite).
"""
struct GlorotUniform <: Initializer end

function (::GlorotUniform)(rng, x::AbstractVecOrMat{T}) where T
    rand!(rng, x)
    x .= sqrt(T(24.0) / sum(size(x))) * (x .- T(0.5)) 
end

const DefaultInitializer = GlorotUniform