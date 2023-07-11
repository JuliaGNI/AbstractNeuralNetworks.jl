
abstract type Model end


"""
    initialparameters

    Returns the initial parameters of a model, i.e., a layer or chain.

```
initialparameters(rng::AbstractRNG, backend::Backend, ::Type{T}, model::Model; init::Callable = default_initializer())
initialparameters(rng::AbstractRNG, ::Type{T}, model::Model; init::Callable = default_initializer())
initialparameters(rng::AbstractRNG, x::AbstractArray, model::Model; init::Callable = default_initializer())
```

The `init!` function must have the following signature:
```
init!(rng::AbstractRNG, x::AbstractArray)
```
The `default_initializer()` returns `randn`.

"""
function initialparameters end


_initialparameters_error(model::Model) = error("initialparameters not implemented for model type ", typeof(model))

initialparameters(::AbstractRNG, ::Backend, ::Type, model::Model; kwargs...) = _initialparameters_error(model)
initialparameters(::AbstractRNG, ::AbstractArray, model::Model; kwargs...) = _initialparameters_error(model)

function initialparameters(rng::AbstractRNG, ::Type{T}, model::Model; kwargs...) where {T}
    initialparameters(rng, CPU(), T, model; kwargs...)
end

# initialparameters(args...; kwargs...) = initialparameters(Random.default_rng(), args...; kwargs...)

initialparameters(backend::Backend, ::Type{T}, model::Model; kwargs...) where {T} = initialparameters(Random.default_rng(), backend, T, model; kwargs...)
initialparameters(x::AbstractArray, model::Model; kwargs...) = initialparameters(Random.default_rng(), x, model; kwargs...)
initialparameters(::Type{T}, model::Model; kwargs...) where {T} = initialparameters(Random.default_rng(), T, model; kwargs...)
