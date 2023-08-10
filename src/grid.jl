
struct GridCell{M, N, CT <: AbstractMatrix} <: Model
    cells::CT
    function GridCell(cells)
        m,n = size(cells)
        new{m, n, typeof(cells)}(cells)
    end
end

(model::GridCell)(x, st, ps) = applygrid(model, x, st, ps)

@inline cells(g::GridCell) = g.cells
@inline cell(g::GridCell, i, j) = g.cells[i, j]
@inline cell(g::GridCell, i) = g.cells[i]

Base.length(g::GridCell) = length(g.cells)
Base.size(g::GridCell) = size(g.cells)
Base.iterate(g::GridCell, i=1) = i > length(g) ? nothing : (cell(g, i), i+1)

@generated function applygrid(gridcell::GridCell{M,N}, x::AbstractArray, st::AbstractArray, ps::Tuple) where {M,N}
    x_symbols = vcat(reshape([:(x[$i]) for i in 1:N], (1,N)), [gensym() for _ in 1:M, _ in 1:N])
    st_symbols = hcat([:(st[$i]) for i in 1:M], [gensym() for _ in 1:M, _ in 1:N])
    calls = vcat([:(($(x_symbols[j+1,i]), $(st_symbols[j,i+1])) = AbstractNeuralNetworks.cell(gridcell, $j, $i)($(x_symbols[j,i]), $(st_symbols[j,i]), ps[$j][$i])) for j in 1:M, i in 1:N]...)
    push!(calls, :(return $(x_symbols[M+1,N])))
    return Expr(:block, calls...)
end 
#
#
function initialparameters(backend::Backend, ::Type{T}, model::GridCell; kwargs...) where {T}
    Tuple(initialparameters(backend, T, cell; kwargs...) for cell in model)
end

function update!(grid::GridCell, params::Tuple, grad::Tuple, η::AbstractFloat)
    for (cell, θ, dθ) in zip(grid, params, grad)
        update!(cell, θ, dθ, η)
    end
end


