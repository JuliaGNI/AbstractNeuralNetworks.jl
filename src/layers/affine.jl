
const Affine{M, N} = Dense{M, N, true, <: IdentityActivation}

Affine(m, n; kwargs...) = Dense(m, n, IdentityActivation(); use_bias = true, kwargs...)

function (layer::Affine)(y::AbstractArray, x::AbstractArray, ps::NamedTuple)
    mul!(y, ps.W, x)
    add!(y, ps.b)
end

function (layer::Affine)(x::AbstractArray, ps::NamedTuple)
    ps.W * x .+ ps.b
end
