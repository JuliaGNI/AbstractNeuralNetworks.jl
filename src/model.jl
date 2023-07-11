
abstract type Model end


"""
    initialparameters

    Returns the initial parameters of a model, i.e., a layer or chain.

```
initialparameters(backend::Backend, ::Type{T}, model::Model; init::Callable = default_initializer(), rng::AbstractRNG = Random.default_rng())
initialparameters(::Type{T}, model::Model; init::Callable = default_initializer(), rng::AbstractRNG = Random.default_rng())
```

The `init!` function must have the following signature:
```
init!(rng::AbstractRNG, x::AbstractArray)
```
The `default_initializer()` returns `randn!`.

"""
function initialparameters end

initialparameters(::Backend, ::Type, model::Model; kwargs...) = error("initialparameters not implemented for model type ", typeof(model))
initialparameters(::Type{T}, model::Model; kwargs...) where {T} = initialparameters(CPU(), T, model; kwargs...)

initialparameters(rng::AbstractRNG, backend::Backend, ::Type{T}, model::Model; kwargs...) where {T} = initialparameters(backend, T, model; rng = rng, kwargs...)
initialparameters(rng::AbstractRNG, ::Type{T}, model::Model; kwargs...) where {T} = initialparameters(T, model; rng = rng, kwargs...)
