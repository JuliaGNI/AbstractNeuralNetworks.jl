using AbstractNeuralNetworks 
using Test 
import Random 

function test_different_cpu_constructors(T₁::Type{T}) where T <: Number
    model = Chain(Dense(4, 5, tanh), Linear(5, 4))
    Random.seed!(123)
    nn₁  = NeuralNetwork(model, T₁, CPU())
    Random.seed!(123)
    nn₂  = NeuralNetwork(model, T₁)
    Random.seed!(123)
    nn₃  = NeuralNetwork(CPU(), T₁, model)
    Random.seed!(123)
    nn₄  = NeuralNetwork(CPU(), model, T₁)
    Random.seed!(123)
    nn₅  = NeuralNetwork(T₁, CPU(), model)
    Random.seed!(123)
    nn₆  = NeuralNetwork(T₁, model, CPU())
    Random.seed!(123)
    nn₇  = NeuralNetwork(model, CPU(), T₁)
    Random.seed!(123)
    nn₈  = NeuralNetwork(T₁, model)
    Random.seed!(123)
    nn₉  = T₁ == Float64 ? NeuralNetwork(CPU(), model) : nn₇
    Random.seed!(123)
    nn₁₀ = T₁ == Float64 ? NeuralNetwork(model, CPU()) : nn₇
    Random.seed!(123)
    nn₁₁ = T₁ == Float64 ? NeuralNetwork(model) : nn₈

    @test nn₁.params == nn₂.params == nn₃.params == nn₄.params == nn₅.params == nn₆.params == nn₇.params == nn₈.params == nn₉.params == nn₁₀.params == nn₁₁.params
end

test_different_cpu_constructors(Float32)
test_different_cpu_constructors(Float64)
test_different_cpu_constructors(Float16)