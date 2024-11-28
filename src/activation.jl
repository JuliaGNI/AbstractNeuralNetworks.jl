
abstract type Activation end
abstract type ScalarActivation <: Activation end
abstract type VectorActivation <: Activation end

struct IdentityActivation <: ScalarActivation end

(::IdentityActivation)(x) = x

struct SigmoidActivation <: ScalarActivation end

(::SigmoidActivation)(x, λ = 1) = 1/(1+exp(-λ*x))


struct GenericActivation{ST <: Base.Callable} <: ScalarActivation 
    σ::ST
    function GenericActivation(σ)
        # TODO: Check if sigma takes scalar arguments and return scalar
        new{typeof(σ)}(σ)
    end
end

(act::GenericActivation)(args...) = act.σ(args...)

Activation(σ::Base.Callable) = GenericActivation(σ)
Activation(σ::Activation) = σ
