"""
    CPUStatic <: KernelAbstractions.Backend

An additional backend that specifies allocation of [static arrays](https://github.com/JuliaArrays/StaticArrays.jl).
"""
struct CPUStatic <: KernelAbstractions.Backend end

function KernelAbstractions.ones(::CPUStatic, ::Type{T}, dims::Tuple) where T
    ones(MArray{Tuple{dims...}, T})
end

function KernelAbstractions.zeros(::CPUStatic, ::Type{T}, dims::Tuple) where T
    zeros(MArray{Tuple{dims...}, T})
end

function KernelAbstractions.allocate(::CPUStatic, ::Type{T}, dims::Tuple) where T
    similar(MArray{Tuple{dims...}, T})
end

_statify(::AbstractArray) = error("Only dense CPU arrays can be made static!")

function _statify(x::Array)
    MArray{Tuple{size(x)...}}(x)
end

function _statify(ps::NamedTuple)
    _keys = keys(ps)
    NamedTuple{_keys}(_statify.(values(ps)))
end

function _statify(ps::NeuralNetworkParameters)
    _keys = keys(ps)
    NeuralNetworkParameters{_keys}(_statify.(values(ps)))
end

function KernelAbstractions.copyto!(::CPUStatic, x::MArray, y::AbstractArray)
    copyto!(x, y)
    nothing
end

function KernelAbstractions.copyto!(::CPUStatic, x::MArray, y::AbstractGPUArray)
    copyto!(x, Array(y))
    nothing
end

#type pyracy!
function KernelAbstractions.get_backend(::MArray)
    CPUStatic()
end

function KernelAbstractions.get_backend(::StaticArray)
    error("You should only use mutable static arrays.")
end