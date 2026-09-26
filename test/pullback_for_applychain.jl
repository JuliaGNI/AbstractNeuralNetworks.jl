using Zygote
using LinearAlgebra: dot, norm
using AbstractNeuralNetworks
using AbstractNeuralNetworks: applychain
using NeuralNetworkParameters
using Test
import Random

Random.seed!(123)

nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))

gradient_result = Zygote.gradient(ps -> norm(applychain(nn.model.layers, rand(10), ps)), nn.params)[1]
pullback_result = Zygote.pullback(applychain, nn.model.layers, rand(10), nn.params)[2](rand(10))[3]

@test typeof(gradient_result) <: NetworkParameters
@test typeof(pullback_result) <: NetworkParameters

# A symmetric matrix that stores its lower triangle, row by row. Its storage is not its interface, so
# the gradient with respect to the storage, ∂L/∂S, counts each stored off-diagonal entry twice: the
# `storage_gradient` of a natural cotangent `G` is the lower triangle of `G + Gᵀ`, the diagonal
# counted once.
struct Sym{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int
end

Base.size(A::Sym) = (A.n, A.n)

function Base.getindex(A::Sym, i::Int, j::Int)
    p, q = max(i, j), min(i, j)
    A.S[((p - 1) * p) ÷ 2 + q]
end

NeuralNetworkParameters.freeparameters(A::Sym) = A.S
NeuralNetworkParameters.rebuild(A::Sym, data) = Sym(data, A.n)

_lower(f, n) = [f(i, j) for i in 1:n for j in 1:i]

function NeuralNetworkParameters.storage_gradient(A::Sym, G::AbstractMatrix)
    Sym(_lower((i, j) -> i == j ? G[i, i] : G[i, j] + G[j, i], A.n), A.n)
end

# The central difference of `f` at `v` in every direction of `v`.
function central_difference(f, v::AbstractVector{T}) where {T}
    h = cbrt(eps(T))
    map(eachindex(v)) do i
        e = zero(v)
        e[i] = h
        (f(v + e) - f(v - e)) / (2h)
    end
end

# A top-level `Zygote.pullback(applychain, layers, x, ps)` gives the gradient of each leaf with respect
# to its storage, as the gradient of a loss with respect to a `NetworkParameters` does.
@testset "applychain pullback gives the storage gradient ($T)" for T in (Float32, Float64)
    layers = (Dense(3, 3, tanh), Dense(3, 2, tanh))
    S = T[0.3, -0.2, 0.5, 0.1, 0.4, -0.6]
    W₂ = T[0.2 -0.1 0.3; -0.4 0.5 0.1]
    b₁ = T[0.1, -0.2, 0.3]
    b₂ = T[-0.1, 0.2]
    x = T[0.7, -0.3, 0.2]
    ȳ = T[0.9, -0.5]

    parameters(S, W₂) = NetworkParameters((
        L1 = (W = Sym(S, 3), b = b₁), L2 = (W = W₂, b = b₂)))

    _, pb = Zygote.pullback(applychain, layers, x, parameters(S, W₂))
    p̄ = pb(ȳ)[3]

    @test p̄ isa NetworkParameters
    @test p̄.L1.W isa Sym{T}

    rtol = sqrt(sqrt(eps(T)))
    @test p̄.L1.W.S ≈
          central_difference(s -> dot(ȳ, applychain(layers, x, parameters(s, W₂))), S) rtol=rtol
    @test vec(p̄.L2.W) ≈ central_difference(
        w -> dot(ȳ, applychain(layers, x, parameters(S, reshape(w, size(W₂))))), vec(W₂)) rtol=rtol
end

# A layer without parameters. A chain of such layers gives the parameters no cotangent at all.
struct NoParameters end
(::NoParameters)(x, ps) = tanh.(x)

@testset "applychain pullback without a parameter cotangent" begin
    x = [0.1, 0.2]
    _, pb = Zygote.pullback(applychain, (NoParameters(), NoParameters()), x,
        NetworkParameters((L1 = NamedTuple(), L2 = NamedTuple())))
    _, x̄, p̄ = pb(ones(2))

    @test p̄ === nothing
    @test x̄ ≈ (1 .- tanh.(tanh.(x)) .^ 2) .* (1 .- tanh.(x) .^ 2)
end
