using AbstractNeuralNetworks
using LinearAlgebra: norm
using Test
using Zygote

using AbstractNeuralNetworks: _norm, _diff, _add, _compute_loss, applychain,
                              ArrayNamedTuple, ArrayOrNamedTuple

# The `(:q, :p)` pair that `QPT`/`QPTOAT` used to hardcode, and a three-field tuple that has nothing
# to do with Hamiltonian phase space. The latter is the point: before #31 it could be evaluated but
# not differentiated, because only the `(:q, :p)` methods avoided the `apply` generator that
# `ChainRulesCore.ProjectTo` chokes on.

qp  = (q = [1.0, 2.0, 3.0], p = [4.0, 5.0, 6.0])
qp2 = (q = [0.5, 1.0, 1.5], p = [2.0, 2.5, 3.0])
ab  = (a = [1.0, 2.0], b = [3.0, 4.0], c = [5.0, 6.0])
ab2 = (a = [0.5, 1.0], b = [1.5, 2.0], c = [2.5, 3.0])


# --- the type aliases ---------------------------------------------------------------------------

@test qp isa ArrayNamedTuple
@test ab isa ArrayNamedTuple
@test qp isa ArrayNamedTuple{Float64}
@test rand(3) isa ArrayOrNamedTuple
@test qp isa ArrayOrNamedTuple
# not every `NamedTuple` -- the values have to be arrays, and of one element type
@test !((q = 1.0, p = 2.0) isa ArrayNamedTuple)
@test !((q = rand(3), p = rand(Float32, 3)) isa ArrayNamedTuple{Float64})


# --- values are unchanged from the deleted `(:q, :p)` methods -----------------------------------
# `GeometricMachineLearning` imports `_compute_loss` and drives it with `(q, p)` data, so these
# have to agree to the last bit with what the specialisations computed.

@test _norm(qp) ≈ (norm(qp.q) + norm(qp.p)) / √2
@test _diff(qp, qp2) == (q = qp.q - qp2.q, p = qp.p - qp2.p)
@test _add(qp, qp2) == (q = qp.q + qp2.q, p = qp.p + qp2.p)
let v = rand(3); @test _norm(v) == norm(v) end

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
