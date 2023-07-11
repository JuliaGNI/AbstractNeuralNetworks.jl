
abstract type AbstractInitializer end

const Initializer = Union{AbstractInitializer, Base.Callable}

struct ZeroInitializer <: AbstractInitializer end
(::ZeroInitializer)(_, x) = x .= 0

struct OneInitializer <: AbstractInitializer end
(::OneInitializer)(_, x) = x .= 1

default_initializer() = randn!
