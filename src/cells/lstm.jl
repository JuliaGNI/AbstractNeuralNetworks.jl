
struct LSTM{M, N, O, P, AOT, AST} <: AbstractExplicitCell{M,N,O,P}
    σ₀::AOT
    σ₋₁::AST

    function LSTM(m, n, σ₀ = SigmoidActivation(), σ₋₁ = tanh)
        new{m, 2*n, n, 2*n, typeof(σ₀), typeof(σ₋₁)}(σ₀, σ₋₁)
    end
end

function (cell::LSTM{M, N, O, P})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,O,P}
    h = st[1:O]
    c = st[O+1:end]
    f = cell.σ₀.(ps.Wfₓ * x + ps.Wfₕ * h + ps.bf)
    i = cell.σ₀.(ps.Wᵢₓ * x + ps.Wᵢₕ * h + ps.bᵢ)
    o = cell.σ₀.(ps.Wₒₓ * x + ps.Wₒₕ * h + ps.bₒ)
    nc = @. f * c + i * cell.σ₋₁(h)
    nh = @. o * cell.σ₋₁(nc)
    return (nh, [nh..., nc...])
end


function initialparameters(backend::Backend, ::Type{T}, cell::LSTM{M, N, O, P}; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng()) where {M,N,O,P,T}
    Wfₓ = KernelAbstractions.zeros(backend, T, O, M)
    Wfₕ = KernelAbstractions.zeros(backend, T, O, O)
    Wᵢₓ = KernelAbstractions.zeros(backend, T, O, M)
    Wᵢₕ = KernelAbstractions.zeros(backend, T, O, O)
    Wₒₓ = KernelAbstractions.zeros(backend, T, O, M)
    Wₒₕ = KernelAbstractions.zeros(backend, T, O, O)
    bf = KernelAbstractions.zeros(backend, T, O)
    bᵢ = KernelAbstractions.zeros(backend, T, O)
    bₒ = KernelAbstractions.zeros(backend, T, O)
    init(rng, Wfₓ)
    init(rng, Wfₕ)
    init(rng, Wᵢₓ)
    init(rng, Wᵢₕ)
    init(rng, Wₒₓ)
    init(rng, Wₒₕ)
    init(rng, bf)
    init(rng, bᵢ)
    init(rng, bₒ)
    (Wfₓ = Wfₓ, Wfₕ = Wfₕ, Wᵢₓ = Wᵢₓ, Wᵢₕ = Wᵢₕ, Wₒₓ = Wₒₓ, Wₒₕ = Wₒₕ, bf = bf, bᵢ = bᵢ, bₒ = bₒ)
end

function update!(::LSTM, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)
    for obj in keys(θ)
        θ[obj] .+= η * dθ[obj]
    end
end
