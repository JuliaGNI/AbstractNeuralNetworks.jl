
struct Dense{M, N, ST} <: AbstractExplicitLayer{M, N}
    σ::ST

    Dense(m, n, σ) = new{m, n, typeof(σ)}(σ)
end

function (layer::Dense)(x::AbstractArray, ps::NamedTuple)
    layer.σ.(ps.W * x .+ ps.b)
end

function (layer::Dense)(y::AbstractArray, x::AbstractArray, ps::NamedTuple)
    mul!(y, ps.W, x)
    add!(y, ps.b)
    y .= layer.σ.(y)
end


function initialparameters(rng::AbstractRNG, W, b, ::Dense; init::Callable = default_initializer())
    init(rng, W)
    init(rng, b)
    (W = W, b = b)
end

function initialparameters(rng::AbstractRNG, backend::Backend, ::Type{T}, layer::Dense{M,N}; kwargs...) where {M,N,T}
    W = KernelAbstractions.zeros(backend, T, N, M)
    b = KernelAbstractions.zeros(backend, T, N)
    initialparameters(rng, W, b, layer; kwargs...)
end

function initialparameters(rng::AbstractRNG, x::AbstractArray, layer::Dense{M,N}; kwargs...) where {M,N}
    W = similar(x, N, M)
    b = similar(x, N)
    initialparameters(rng, W, b, layer; kwargs...)
end
