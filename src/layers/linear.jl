
const Linear{M, N} = Dense{M, N, false, <: IdentityActivation}

Linear(m, n; kwargs...) = Dense(m, n, IdentityActivation();use_bias = false, kwargs...)

function (layer::Linear)(y::AbstractArray, x::AbstractArray, ps::NamedTuple)
    mul!(y, ps.W, x)
end

function (layer::Linear)(x::AbstractArray, ps::NamedTuple)
    ps.W * x
end
