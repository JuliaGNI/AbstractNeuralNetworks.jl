
const Linear{M, N} = Dense{M, N, <: IdentityActivation}

Linear(m, n) = Dense(m, n, IdentityActivation())

function (layer::Linear)(y::AbstractArray, x::AbstractArray, ps::NamedTuple)
    mul!(y, ps.W, x)
    add!(y, ps.b)
end

function (layer::Linear)(x::AbstractArray, ps::NamedTuple)
    ps.W * x .+ ps.b
end
