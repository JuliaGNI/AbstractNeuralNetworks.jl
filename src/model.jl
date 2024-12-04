"""
A supertype for `Chain`, `AbstractCell` etc.
"""
abstract type Model end


"""
    initialparameters

    Returns the initial parameters of a model, i.e., a layer or chain.

```
initialparameters(backend::Backend, ::Type{T}, model::Model; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng())
initialparameters(::Type{T}, model::Model; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng())
```

The `init!` function must have the following signature:
```
init!(rng::AbstractRNG, x::AbstractArray)
```
The `default_initializer()` returns `randn!`.

"""
function initialparameters end

initialparameters(model::Model, ::Backend, ::Type; kwargs...) = error("initialparameters not implemented for model type ", typeof(model))
initialparameters(::Model, backend::Backend; kwargs...) = initialparameters = error("No default type defined for $(backend).")
initialparameters(model::Model, backend::Union{CPU, CPUStatic}; kwargs...) = initialparameters(model, backend, Float64; kwargs...)
initialparameters(model::Model, backend::GPU; kwargs...) = initialparameters(model, backend, Float32; kwargs...) 
initialparameters(model::Model, ::Type{T}; kwargs...) where {T} = initialparameters(model, CPU(), T; kwargs...)

initialparameters(rng::AbstractRNG, model::Model, ::Backend, ::Type; kwargs...) = error("initialparameters not implemented for model type ", typeof(model))
initialparameters(rng::AbstractRNG, ::Model, backend::Backend; kwargs...) = initialparameters = error("No default type defined for $(backend).")
initialparameters(rng::AbstractRNG, model::Model, backend::Union{CPU, CPUStatic}; kwargs...) = initialparameters(model, backend, Float64; rng = rng, kwargs...)
initialparameters(rng::AbstractRNG, model::Model, backend::GPU; kwargs...) = initialparameters(model, backend, Float32; rng = rng, kwargs...) 
initialparameters(rng::AbstractRNG, model::Model, ::Type{T}; kwargs...) where {T} = initialparameters(model, CPU(), T; rng = rng, kwargs...)

function parameterlength end

Base.eachindex(m::Model) = @error "You forgot to define the eachindex function for the model of type "*string(typeof(m))*"!"

update!(model::Model, params::Union{NamedTuple,NeuralNetworkParameters}, grad::Union{NamedTuple,NeuralNetworkParameters}, args...) = update!(model, values(params), values(grad), args...)
