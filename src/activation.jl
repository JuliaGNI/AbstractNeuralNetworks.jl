
abstract type AbstractActivation end
abstract type ScalarActivation <: AbstractActivation end
abstract type VectorActivation <: AbstractActivation end


struct IdentityActivation <: ScalarActivation end

(::IdentityActivation)(x) = x
