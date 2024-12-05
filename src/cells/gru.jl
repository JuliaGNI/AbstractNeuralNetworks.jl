
struct GRU{M, N, O, P, AOT, AST} <: AbstractExplicitCell{M,N,O,P}
    σ₀::AOT
    σ₋₁::AST

    function GRU(m, n, σ₀ = SigmoidActivation(), σ₋₁ = tanh)
        new{m, n, n, n, typeof(σ₀), typeof(σ₋₁)}(σ₀, σ₋₁)
    end
end

function (cell::GRU{M, N, O, P})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,O,P}
    r = cell.σ₀.(ps.Wᵣₓ * x + ps.Wᵣₕ * st + ps.bᵣ)
    u = cell.σ₀.(ps.Wᵤₓ * x + ps.Wᵤₕ * st + ps.bᵤ)
    h = cell.σ₋₁.(ps.Wⱼₓ * x + ps.Wⱼₕ * (r .* st) + ps.bⱼ)
    ns = @. u * st + (1 - u) .* h 
    return (ns, ns)
end


function initialparameters(cell::GRU{M, N, O, P}, backend::NeuralNetworkBackend, ::Type{T}; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng()) where {M,N,O,P,T}
    Wᵣₓ = KernelAbstractions.zeros(backend, T, N, M)
    Wᵣₕ = KernelAbstractions.zeros(backend, T, N, N)
    Wᵤₓ = KernelAbstractions.zeros(backend, T, N, M)
    Wᵤₕ = KernelAbstractions.zeros(backend, T, N, N)
    Wⱼₓ = KernelAbstractions.zeros(backend, T, N, M)
    Wⱼₕ = KernelAbstractions.zeros(backend, T, N, N)
    bᵣ = KernelAbstractions.zeros(backend, T, N)
    bᵤ = KernelAbstractions.zeros(backend, T, N)
    bⱼ = KernelAbstractions.zeros(backend, T, N)
    init(rng, Wᵣₓ)
    init(rng, Wᵣₕ)
    init(rng, Wᵤₓ)
    init(rng, Wᵤₕ)
    init(rng, Wⱼₓ)
    init(rng, Wⱼₕ)
    init(rng, bᵣ)
    init(rng, bᵤ)
    init(rng, bⱼ)
    (Wᵣₓ = Wᵣₓ, Wᵣₕ = Wᵣₕ, Wᵤₓ = Wᵤₓ, Wᵤₕ = Wᵤₕ, Wⱼₓ = Wⱼₓ, Wⱼₕ = Wⱼₕ, bᵣ = bᵣ, bᵤ = bᵤ, bⱼ = bⱼ)
end

function update!(::GRU, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)
    for obj in keys(θ)
        θ[obj] .+= η * dθ[obj]
    end
end
