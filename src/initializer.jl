
abstract type Initializer end

struct ZeroInitializer <: Initializer end
(ZeroInitializer)(_, x) = x .= 0

struct OneInitializer <: Initializer end
(OneInitializer)(_, x) = x .= 1

default_initializer() = randn!
