# Type aliases for what a network can be applied to, and what a loss can be computed over: an array,
# or a `NamedTuple` of arrays. These generalise the `QPT`/`QPTOAT` pair that used to live in
# `losses.jl`, which fixed the keys to `(:q, :p)` -- Hamiltonian phase-space vocabulary that has no
# business in an architecture-agnostic package (issue #31). The shape follows
# `GeometricOptimizers`' `ArrayNamedTuple`.

# Note that this is *not* `Tuple{Vararg{AT}} where {AT <: AbstractArray{T}}`: Julia's diagonal rule
# would make that homogeneous, i.e. it would reject a `NamedTuple` that stores e.g. a
# `StiefelManifold` and an ordinary `Matrix` at the same time. `QPT` did couple both entries to one
# `AT`, so these aliases are strictly wider than what they replace.
const ArrayTuple{T} = Tuple{Vararg{AbstractArray{T}}}

"""
    ArrayNamedTuple{T, S}

A `NamedTuple` with keys `S` whose values are all `AbstractArray{T}`.

!!! warning
    Use this in method signatures, where it dispatches. As a bound on the type parameters of a
    `struct` it is ruinously expensive, because it *couples* the parameters -- inference cannot
    solve `NamedTuple{S, <:Tuple{Vararg{AbstractArray{T}}}}` down to a concrete `NamedTuple`.
"""
const ArrayNamedTuple{T, S} = NamedTuple{S, <:ArrayTuple{T}}

"""
    ArrayOrNamedTuple{T}

Either an `AbstractArray{T}` or an [`ArrayNamedTuple{T}`](@ref) -- the inputs and outputs a
[`Model`](@ref) can be applied to and a [`NetworkLoss`](@ref) computed over.

See the warning on [`ArrayNamedTuple`](@ref) about `struct` type-parameter bounds.
"""
const ArrayOrNamedTuple{T} = Union{AbstractArray{T}, ArrayNamedTuple{T}}
