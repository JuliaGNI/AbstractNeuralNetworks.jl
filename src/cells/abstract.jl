@doc raw"""
    AbstractCell

An `AbstractCell` is a map from $\mathbb{R}^{M}×\mathbb{R}^{N} \rightarrow \mathbb{R}^{O}×\mathbb{R}^{P}$.

Concrete cell types should implement the following functions:

- `initialparameters(backend::Backend, ::Type{T}, cell::AbstractCell; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng())`
- `update!(::AbstractLayer, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)`

and the functors

- `cell(x, st, ps)`
- `cell(z, y, x, st, ps)`

"""
abstract type AbstractCell{M,N,O,P} <: Model end


"""
    apply(cayer::AbstractCell, x, ps)

Simply calls `cell(x, st, ps)`
"""
function apply(cell::AbstractCell, x, st, ps)
    return cell(x, st, ps)
end

"""
    apply!(y, cell::AbstractCell, x, ps)

Simply calls `cell(y, x, ps)`
"""
function apply!(z::AbstractArray, y::AbstractArray, cell::AbstractLayer, x, ps)
    return cell(z, y, x, ps)
end


"""
    AbstractExplicitCell

Abstract supertype for explicit cells.
This type exists mainly for compatibility with Lux.
"""
abstract type AbstractExplicitCell{M,N,O,P} <: AbstractCell{M,N,O,P} end
