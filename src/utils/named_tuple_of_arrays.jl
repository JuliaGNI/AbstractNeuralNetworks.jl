# Type aliases for what a network can be applied to, and what a loss can be computed over: an array,
# or a `NamedTuple` of arrays. These generalise the `QPT`/`QPTOAT` pair that used to live in
# `losses.jl`, which fixed the keys to `(:q, :p)` -- Hamiltonian phase-space vocabulary that has no
# business in an architecture-agnostic package (issue #31).
#
# `NamedTupleOfArrays` and not `ArrayNamedTuple`: the name says what the alias is about, which here is
# a network's *inputs and outputs*. A set of *parameters* is a
# `NeuralNetworkParameters.NetworkParameters` and not an alias at all — an alias for `Base.NamedTuple`
# is a type nobody owns, so a method taking one is a method on a `Base` type. One name, one thing.

# Note that this is *not* `Tuple{Vararg{AT}} where {AT <: AbstractArray{T}}`: Julia's diagonal rule
# would make that homogeneous, i.e. it would reject a `NamedTuple` that stores e.g. a
# `StiefelManifold` and an ordinary `Matrix` at the same time. `QPT` did couple both entries to one
# `AT`, so these aliases are strictly wider than what they replace.
const TupleOfArrays{T} = Tuple{Vararg{AbstractArray{T}}}

"""
    NamedTupleOfArrays{T, S}

A `NamedTuple` with keys `S` whose values are all `AbstractArray{T}`.

!!! warning
    Use this in method signatures, where it dispatches. As a bound on the type parameters of a
    `struct` it is ruinously expensive, because it *couples* the parameters -- inference cannot
    solve `NamedTuple{S, <:Tuple{Vararg{AbstractArray{T}}}}` down to a concrete `NamedTuple`.
"""
const NamedTupleOfArrays{T, S} = NamedTuple{S, <:TupleOfArrays{T}}

"""
    ArrayOrNamedTuple{T}

Either an `AbstractArray{T}` or a [`NamedTupleOfArrays{T}`](@ref) -- the inputs and outputs a
[`Model`](@ref) can be applied to and a [`NetworkLoss`](@ref) computed over.

See the warning on [`NamedTupleOfArrays`](@ref) about `struct` type-parameter bounds.
"""
const ArrayOrNamedTuple{T} = Union{AbstractArray{T}, NamedTupleOfArrays{T}}
