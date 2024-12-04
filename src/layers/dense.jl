struct Dense{M, N, BIAS, ST <: Activation} <: AbstractExplicitLayer{M, N}
    σ::ST
end

function Dense(m, n, σ = tanh; use_bias = true)
    act = Activation(σ)
    Dense{m, n, use_bias, typeof(act)}(act)
end

function (layer::Dense{M,N,true})(x::AbstractArray, ps::NamedTuple) where {M,N}
    layer.σ.(ps.W * x .+ ps.b)
end

function (layer::Dense{M,N,false})(x::AbstractArray, ps::NamedTuple) where {M,N}
    layer.σ.(ps.W * x)
end

function (layer::Dense)(y::AbstractArray, x::AbstractArray, ps::NamedTuple)
    mul!(y, ps.W, x)
    if usebias(layer)
        add!(y, ps.b)
    end
    y .= layer.σ.(y)
end

usebias(::Dense{M, N, BIAS}) where {M, N, BIAS} = BIAS

function initialparameters(rng::AbstractRNG, init::Initializer, ::Dense{M,N,true}, backend::Backend, ::Type{T}) where {M,N,T}
    W = KernelAbstractions.zeros(backend, T, N, M)
    b = KernelAbstractions.zeros(backend, T, N)
    init(rng, W)
    init(rng, b)
    (W = W, b = b)
end

function initialparameters(rng::AbstractRNG, init::Initializer, ::Dense{M,N,false}, backend::Backend, ::Type{T}) where {M,N,T}
    W = KernelAbstractions.zeros(backend, T, N, M)
    init(rng, W)
    (W = W,)
end

function update!(::Dense, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)
    for obj in keys(θ)
        θ[obj] .+= η * dθ[obj]
    end
end

function parameterlength(::Dense{M, N, BIAS}) where {M, N, BIAS}
    BIAS == true ? (M*N + N) : (M*N)
end