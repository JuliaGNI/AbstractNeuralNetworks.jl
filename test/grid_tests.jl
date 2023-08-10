using AbstractNeuralNetworks
using Random
using Test

input = [2,2]
c = Recurrent(2,3,4,5, x->x)


st = [3,4,7]

p = initialparameters(Random.default_rng(), Float64, c; init = OneInitializer())

o, s = c(input, st, p)

input = [[2,2],[2,2]]
st = [[3,4], [7,8]]
c1 = Recurrent(2, 2, 2, 2, tanh)
c2 = Recurrent(2, 2, 2, 2, tanh)
p = initialparameters(Random.default_rng(), Float64, c1; init = OneInitializer())
params = ((p, p),)
g = GridCell( [c1  c2])

@test AbstractNeuralNetworks.cell(g, 1, 1) == c1
@test AbstractNeuralNetworks.cell(g, 1, 2) == c2


@generated function applygrid2(gridcell::GridCell{M,N}, x::AbstractArray, st::AbstractArray, ps::Tuple) where {M,N}
    x_symbols = vcat(reshape([:(x[$i]) for i in 1:N], (1,N)), [gensym() for _ in 1:M, _ in 1:N])
    st_symbols = hcat([:(st[$i]) for i in 1:M], [gensym() for _ in 1:M, _ in 1:N])
    calls = vcat([:(($(x_symbols[j+1,i]), $(st_symbols[j,i+1])) = AbstractNeuralNetworks.cell(gridcell, $j, $i)($(x_symbols[j,i]), $(st_symbols[j,i]), ps[$j][$i])) for j in 1:M, i in 1:N]...)
    push!(calls, :(return $(x_symbols[M+1,N])))
    return Expr(:block, calls...)
end 

applygrid2(g, input, st, params)

g(input, st, params)
