"""
    Architecture
"""
abstract type Architecture end

struct UnknownArchitecture <: Architecture end

dim(arch::Architecture) = @error "You forgot to implement dim for $(typeof(arch))."
