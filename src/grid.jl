
struct GridCell{M, N, CT <: AbstractMatrix, IT <: AbstractArray} <: Model
    cells::CT
    init_st::IT
    function GridCell(cells; init_state = missing)
        m,n = size(cells)
        init_st = ismissing(init_state) ? [zeros(size(cells[i,1])[2]) for i in 1:m] : init_state
        new{m, n, typeof(cells), typeof(init_st)}(cells, init_st)
    end
end

(model::GridCell)(x, st, ps) = applygrid(model, x, st, ps)

@inline cells(g::GridCell) = g.cells
@inline cell(g::GridCell, i, j) = g.cells[i, j]
@inline cell(g::GridCell, i) = g.cells[i]

@inline lines(::GridCell{M, N}) where {M,N} = M
@inline rows(::GridCell{M, N}) where {M,N} = N

Base.length(g::GridCell) = length(g.cells)
Base.size(g::GridCell) = size(g.cells)
Base.iterate(g::GridCell, i=1) = i > length(g) ? nothing : (cell(g, i), i+1)
Base.eachindex(g::GridCell) = Iterators.product(1:lines(g), 1:rows(g))

@generated function applygrid(gridcell::GridCell{M,N}, x::AbstractArray, st::AbstractArray, ps::Matrix) where {M,N}
    x_symbols = vcat(reshape([:(x[$i]) for i in 1:N], (1,N)), [gensym() for _ in 1:M, _ in 1:N])
    st_symbols = hcat([:(st[$i]) for i in 1:M], [gensym() for _ in 1:M, _ in 1:N])
    calls = vcat([:(($(x_symbols[j+1,i]), $(st_symbols[j,i+1])) = AbstractNeuralNetworks.cell(gridcell, $j, $i)($(x_symbols[j,i]), $(st_symbols[j,i]), ps[$j,$i])) for j in 1:M, i in 1:N]...)
    push!(calls, :(return $(x_symbols[M+1,N])))
    return Expr(:block, calls...)
end 

function initialparameters(backend::Backend, ::Type{T}, gridcell::GridCell; kwargs...) where {T}
    M,N = size(gridcell)
    [initialparameters(backend, T, cell(gridcell, i, j); kwargs...) for i in 1:M, j in 1:N]
end

function update!(grid::GridCell, params::Matrix, grad::Matrix, η::AbstractFloat)
    for (cell, θ, dθ) in zip(grid, params, grad)
        update!(cell, θ, dθ, η)
    end
end


