const Linear{M, N, USEBIAS} = Dense{M, N, USEBIAS, <: IdentityActivation}

Linear(m, n; kwargs...) = Dense(m, n, IdentityActivation(); kwargs...)

function (layer::Linear)(y::AbstractArray, x::AbstractArray, ps::NamedTuple)
    mul!(y, ps.W, x)
    add!(y, ps.b)
end

function (layer::Linear)(x::AbstractArray, ps::NamedTuple)
    ps.W * x .+ ps.b
end
