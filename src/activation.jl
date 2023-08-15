
abstract type AbstractActivation end
abstract type ScalarActivation <: AbstractActivation end
abstract type VectorActivation <: AbstractActivation end

const Activation = Union{AbstractActivation, Base.Callable}

struct IdentityActivation <: ScalarActivation end

(::IdentityActivation)(x) = x

struct SigmoidActivation <: ScalarActivation end

(::SigmoidActivation)(x, λ = 1) = 1/(1+exp(-λ*x))
