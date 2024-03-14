using AbstractNeuralNetworks 
using Test 
import Random 

function test_different_cpu_constructors(::Type{T}) where T <: Number
    model = Chain(Dense(4, 5, tanh), Linear(5, 4))
    Random.seed!(123)
    nn1 = NeuralNetwork(model, CPU(), T)
    Random.seed!(123)
    nn2  = NeuralNetwork(model, T)
    Random.seed!(123)
    nn3 = T == Float64 ? NeuralNetwork(model, CPU()) : nn2
    Random.seed!(123)
    nn4 = T == Float64 ? NeuralNetwork(model) : nn3

    @test nn1.params == nn2.params == nn3.params == nn4.params
end

test_different_cpu_constructors(Float32)
test_different_cpu_constructors(Float64)
test_different_cpu_constructors(Float16)