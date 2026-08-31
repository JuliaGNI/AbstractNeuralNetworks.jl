using AbstractNeuralNetworks
using LinearAlgebra: norm
using Test
using Zygote

using AbstractNeuralNetworks: _norm, _diff, _add, _compute_loss, applychain,
                              NamedTupleOfArrays, ArrayOrNamedTuple

# The `(:q, :p)` pair that `QPT`/`QPTOAT` used to hardcode, and a three-field tuple that has nothing
# to do with Hamiltonian phase space. The latter is the point: before #31 it could be evaluated but
# not differentiated, because only the `(:q, :p)` methods avoided the `apply` generator that
# `ChainRulesCore.ProjectTo` chokes on.

qp = (q = [1.0, 2.0, 3.0], p = [4.0, 5.0, 6.0])
qp2 = (q = [0.5, 1.0, 1.5], p = [2.0, 2.5, 3.0])
ab = (a = [1.0, 2.0], b = [3.0, 4.0], c = [5.0, 6.0])
ab2 = (a = [0.5, 1.0], b = [1.5, 2.0], c = [2.5, 3.0])

# --- the type aliases ---------------------------------------------------------------------------

@test qp isa NamedTupleOfArrays
@test ab isa NamedTupleOfArrays
@test qp isa NamedTupleOfArrays{Float64}
@test rand(3) isa ArrayOrNamedTuple
@test qp isa ArrayOrNamedTuple
# not every `NamedTuple` -- the values have to be arrays, and of one element type
@test !((q = 1.0, p = 2.0) isa NamedTupleOfArrays)
@test !((q = rand(3), p = rand(Float32, 3)) isa NamedTupleOfArrays{Float64})

# --- values are unchanged from the deleted `(:q, :p)` methods -----------------------------------
# `GeometricMachineLearning` imports `_compute_loss` and drives it with `(q, p)` data, so these
# have to agree to the last bit with what the specialisations computed.

@test _norm(qp) ≈ (norm(qp.q) + norm(qp.p)) / √2
@test _diff(qp, qp2) == (q = qp.q - qp2.q, p = qp.p - qp2.p)
@test _add(qp, qp2) == (q = qp.q + qp2.q, p = qp.p + qp2.p)
let v = rand(3)
    @test _norm(v) == norm(v)
end

# and the same formulas generalise off `(:q, :p)`
@test _norm(ab) ≈ (norm(ab.a) + norm(ab.b) + norm(ab.c)) / √3
@test _diff(ab, ab2) == (a = ab.a - ab2.a, b = ab.b - ab2.b, c = ab.c - ab2.c)

# mismatched keys are still rejected
@test_throws AssertionError _diff(qp, ab)

# --- the regression: gradients through the generic path -----------------------------------------

for (input, output) in ((qp, qp2), (ab, ab2))
    g = Zygote.gradient(x -> _compute_loss(x, output), input)[1]
    @test keys(g) == keys(input)
    @test all(v -> v isa AbstractArray, values(g))
end

@test_nowarn Zygote.gradient(x -> _compute_loss(x, rand(3)), rand(3))

# --- `applychain` is no longer restricted to `(:q, :p)` -----------------------------------------
# `Dense` only takes arrays, so a layer that consumes something else is needed to reach the widened
# dispatch at all. `x` is now untyped, so a `Tuple` has to get through as well -- that is the case
# `GeometricMachineLearning` had to pirate `applychain` for.

struct SumFields <: AbstractNeuralNetworks.AbstractExplicitLayer{1, 1} end
(::SumFields)(x::NamedTuple, ::NamedTuple) = sum(values(x))
(::SumFields)(x::Tuple, ::NamedTuple) = sum(x)

@test applychain((SumFields(),), ab, (NamedTuple(),)) == ab.a + ab.b + ab.c
@test applychain((SumFields(),), qp, (NamedTuple(),)) == qp.q + qp.p
@test applychain((SumFields(),), ([1.0, 2.0], [3.0, 4.0]), (NamedTuple(),)) == [4.0, 6.0]

# --- the public loss and pullback entry points ---------------------------------------------------
# These are the surface `QPTOAT` used to type, so they should be exercised at both an array and a
# `NamedTuple`, and through all three of the ways a loss can be called.

nn = NeuralNetwork(Chain(Dense(3, 3, tanh)), Float64)
loss = FeedForwardLoss()
input, output = rand(3), rand(3)

@test loss(nn, input, output) == loss(nn.model, nn.params, input, output)
@test loss(nn, input, output) == _compute_loss(nn.model, nn.params, input, output)
@test loss(nn, input, output) ≈ norm(output - nn(input)) / norm(output)
@test loss(nn, input, input) ≈ _norm(_diff(nn(input), input)) / _norm(input)

# a `NetworkLoss` with no functor of its own falls back to a message naming the type
struct UnimplementedLoss <: AbstractNeuralNetworks.NetworkLoss end
@test_throws "Functor not defined for `NetworkLoss` of type" UnimplementedLoss()(
    nn.model, nn.params, input, output)
@test_throws "Functor not defined for `NetworkLoss` of type" UnimplementedLoss()(nn, input, output)

# likewise the two `AbstractPullback` extension points documented on the abstract type
struct UnimplementedPullback <: AbstractNeuralNetworks.AbstractPullback{FeedForwardLoss} end
pb = UnimplementedPullback()
@test_throws "Pullback not implemented for input-output pair!" pb(nn.params, nn.model, (
    input, output))
@test_throws "Pullback not implemented for single input!" pb(nn.params, nn.model, input)
@test_throws "Pullback not implemented for single input!" pb(nn.params, nn.model, qp)
