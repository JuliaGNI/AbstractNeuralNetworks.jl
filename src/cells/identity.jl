
struct IdentityCell{M, N, O, P} <: AbstractExplicitCell{M,N,O,P}
    IdentityCell() = new{Any, Any, Any, Any}()
end

function (cell::IdentityCell{M, N, O, P})(x::AbstractArray, st::AbstractArray, ps::NamedTuple) where {M,N,O,P}
    return (x, st)
end

function initialparameters(cell::IdentityCell{M, N, O, P}, backend::Backend, ::Type{T}; init::Initializer = default_initializer(), rng::AbstractRNG = Random.default_rng()) where {M,N,O, P, T}
    NamedTuple()
end

function update!(::IdentityCell, θ::NamedTuple, dθ::NamedTuple, η::AbstractFloat)
    for obj in keys(θ)
        θ[obj] .+= η * dθ[obj]
    end
end
