
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



function initialparameters(rng::AbstractRNG, backend::Backend, ::Type{T}, layer::Dense{M,N}; init::Callable = default_initializer()) where {M,N,T}
    W = KernelAbstractions.zeros(backend, T, N, M)
    b = KernelAbstractions.zeros(backend, T, N)
    init(rng, W)
    init(rng, b)
    (W = W, b = b)
end
