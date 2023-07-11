
abstract type AbstractActivationFunction end

struct IdentityActivation <: AbstractActivationFunction end

(::IdentityActivation)(x) = x
