
struct Recurrent{M, N, O, P, BIAS, AOT, AST} <: AbstractExplicitCell{M,N,O,P}
    σₒ::AOT
    σₛ::AST

    Recurrent(m, n, o, p, σₒ, σₛ = σₒ; use_bias = true) = new{m, n, o, p, use_bias, typeof(σₒ), typeof(σₛ)}(σₒ, σₛ)
end

function (cell::Recurrent{M, N, O, P, true})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,O,P}
    s = cell.σₛ.(ps.Wₛₛ * st + ps.Wₛₓ * x + ps.bₛ)
    y = cell.σₒ.(ps.Wₒₛ * s + ps.bₒ)
    return (y, s)
end

function (cell::Recurrent{M, N, O, P, false})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,O,P}
    s = cell.σₛ.(ps.Wₛₛ * st + ps.Wₛₓ * x)
    y = cell.σₒ.(ps.Wₒₛ * s)
    return (y, s)
end

function (cell::Recurrent{M, N, 0, P, true})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,P}
    s = cell.σₛ.(ps.Wₛₛ * st + ps.Wₛₓ * x + ps.bₛ)
    return (nothing, s)
end

function (cell::Recurrent{M, N, 0, P, false})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,P}
    s = cell.σₛ.(ps.Wₛₛ * st + ps.Wₛₓ * x)
    return (nothing, s)
end

usebias(::Recurrent{M, N, O, P, BIAS}) where {M, N, O, P, BIAS} = BIAS

function initialparameters(cell::Recurrent{M, N, O, P}, backend::Backend, ::Type{T}; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng()) where {M,N,O,P,T}
    Wₛₛ = KernelAbstractions.zeros(backend, T, P, N)
    Wₛₓ = KernelAbstractions.zeros(backend, T, P, M)
    Wₒₛ = KernelAbstractions.zeros(backend, T, O, P)
    bₛ = KernelAbstractions.zeros(backend, T, P)
    bₒ = KernelAbstractions.zeros(backend, T, O)
    init(rng, Wₛₛ)
    init(rng, Wₛₓ)
    init(rng, Wₒₛ)
    init(rng, bₛ)
    init(rng, bₒ)
    (Wₛₛ = Wₛₛ, Wₛₓ = Wₛₓ, Wₒₛ = Wₒₛ, bₛ = bₛ, bₒ = bₒ)
end

function initialparameters(cell::Recurrent{M, N, 0, P}, backend::Backend, ::Type{T}; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng()) where {M,N,P,T}
    Wₛₛ = KernelAbstractions.zeros(backend, T, P, N)
    Wₛₓ = KernelAbstractions.zeros(backend, T, P, M)
    bₛ = KernelAbstractions.zeros(backend, T, P)
    init(rng, Wₛₛ)
    init(rng, Wₛₓ)
    init(rng, bₛ)
    (Wₛₛ = Wₛₛ, Wₛₓ = Wₛₓ, bₛ = bₛ)
end


function update!(::Recurrent, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)
    for obj in keys(θ)
        θ[obj] .+= η * dθ[obj]
    end
end
